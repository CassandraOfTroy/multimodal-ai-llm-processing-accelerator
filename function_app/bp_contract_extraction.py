import json
import logging
import os
from typing import Optional

import azure.functions as func
import jellyfish
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from haystack import Document
from openai import AzureOpenAI
from pydantic import BaseModel, Field
from src.components.doc_intelligence import (
    VALID_DI_PREBUILT_READ_LAYOUT_MIME_TYPES,
    DefaultDocumentFigureProcessor,
    DefaultDocumentPageProcessor,
    DocumentIntelligenceProcessor,
    convert_processed_di_docs_to_openai_message,
)
from src.helpers.common import MeasureRunTime
from src.helpers.data_loading import load_visual_obj_bytes_to_pil_imgs_dict
from src.helpers.image import (
    draw_polygon_on_pil_img,
    flat_poly_list_to_poly_dict_list,
    pil_img_to_base64_bytes,
    resize_img_by_max,
    scale_flat_poly_list,
)
from src.result_enrichment.common import merge_confidence_scores
from src.result_enrichment.doc_intelligence import (
    find_matching_di_lines,
    find_matching_di_words,
)

load_dotenv()

bp_skoda_lease_extraction = func.Blueprint()

# Load environment variables
DOC_INTEL_ENDPOINT = os.getenv("DOC_INTEL_ENDPOINT")
DOC_INTEL_API_KEY = os.getenv("DOC_INTEL_API_KEY")
AOAI_ENDPOINT = os.getenv("AOAI_ENDPOINT")
AOAI_LLM_DEPLOYMENT = os.getenv("AOAI_LLM_DEPLOYMENT")
AOAI_API_KEY = os.getenv("AOAI_API_KEY")

# Set confidence threshold
MIN_CONFIDENCE_SCORE = 0.8
FUNCTION_ROUTE = "skoda_lease_extraction"

# Initialize clients
di_client = DocumentIntelligenceClient(
    endpoint=DOC_INTEL_ENDPOINT,
    credential=AzureKeyCredential(DOC_INTEL_API_KEY),
    api_version="2024-07-31-preview",
)
aoai_client = AzureOpenAI(
    azure_endpoint=AOAI_ENDPOINT,
    azure_deployment=AOAI_LLM_DEPLOYMENT,
    api_key=AOAI_API_KEY,
    api_version="2024-06-01",
    timeout=30,
    max_retries=0,
)

# Configure document processor
doc_intel_result_processor = DocumentIntelligenceProcessor(
    page_processor=DefaultDocumentPageProcessor(page_img_order="after"),
    figure_processor=DefaultDocumentFigureProcessor(output_figure_img=False),
)

class LLMExtractedFieldsModel(BaseModel):
    """Schema for Škoda lease document extraction"""
    vehicle_model: str = Field(
        description="The complete vehicle model name including engine and transmission",
        examples=["Skoda Octavia Combi Tour 2,0 TDI 110 kW 7-Gang-DSG"]
    )
    contract_duration_months: str = Field(
        description="Duration of the lease contract in months",
        examples=["36"]
    )
    annual_mileage: str = Field(
        description="Annual mileage allowance in kilometers",
        examples=["30000"]
    )
    monthly_lease_rate: str = Field(
        description="Monthly lease rate in EUR without VAT",
        examples=["437.49"]
    )
    vehicle_base_price: str = Field(
        description="Base price of the vehicle in EUR",
        examples=["34596.64"]
    )
    color_trim: str = Field(
        description="Vehicle color and interior trim specification",
        examples=["Moon-Weiß Perleffekt, Lounge (Microfaser-Lederausstattung Schwarz)"]
    )
    delivery_date: str = Field(
        description="Expected delivery date (Month.Year)",
        examples=["05.2025"]
    )
    extra_km_rate: str = Field(
        description="Cost per extra kilometer in EUR cents",
        examples=["7.70"]
    )
    under_km_credit: str = Field(
        description="Credit per under-driven kilometer in EUR cents",
        examples=["3.50"]
    )

class FieldWithConfidenceModel(BaseModel):
    """Enriched schema with confidence scores"""
    value: str = Field(description="The extracted value")
    doc_intel_content_matches_count: Optional[int] = Field(
        description="Number of matching content objects found"
    )
    confidence: Optional[float] = Field(
        description="Confidence score for the extracted value"
    )
    normalized_polygons: Optional[list[list[float]]] = Field(
        description="Normalized polygon coordinates for the field location"
    )

class ExtractedFieldsWithConfidenceModel(BaseModel):
    """Complete model with confidence scores"""
    vehicle_model: FieldWithConfidenceModel
    contract_duration_months: FieldWithConfidenceModel
    annual_mileage: FieldWithConfidenceModel
    monthly_lease_rate: FieldWithConfidenceModel
    vehicle_base_price: FieldWithConfidenceModel
    color_trim: FieldWithConfidenceModel
    delivery_date: FieldWithConfidenceModel
    extra_km_rate: FieldWithConfidenceModel
    under_km_credit: FieldWithConfidenceModel

class FunctionResponseModel(BaseModel):
    """Function response model"""
    success: bool = Field(default=False)
    requires_human_review: bool = Field(default=False)
    min_extracted_field_confidence_score: Optional[float] = None
    required_confidence_score: float
    result: Optional[ExtractedFieldsWithConfidenceModel] = None
    error_text: Optional[str] = None
    func_time_taken_secs: Optional[float] = None
    di_extracted_text: Optional[str] = None
    di_raw_response: Optional[dict] = None
    di_time_taken_secs: Optional[float] = None
    llm_input_messages: Optional[list[dict]] = None
    llm_reply_message: Optional[dict] = None
    llm_raw_response: Optional[str] = None
    llm_time_taken_secs: Optional[float] = None
    result_img_with_bboxes: Optional[bytes] = None

# System prompt for GPT
LLM_SYSTEM_PROMPT = """You are a document extraction expert specializing in German automotive lease agreements.
Your task is to extract specific information from Škoda lease documents.
Please extract the information in the following JSON format:

{LLMExtractedFieldsModel.get_prompt_json_example(include_preceding_json_instructions=True)}

Important notes:
- All monetary values should be extracted as numbers without the EUR symbol
- Look for 'Monatliche Leasingrate' for the monthly lease rate
- Vehicle model information is typically found under 'Fahrzeug' or 'Modell'
- Delivery date is found under 'Liefertermin'
- Mileage information is usually specified as 'Fahrleistung' in km/Jahr
"""

@bp_skoda_lease_extraction.route(route=FUNCTION_ROUTE)
def skoda_lease_extraction(req: func.HttpRequest) -> func.HttpResponse:
    """Main function to process Škoda lease documents"""
    logging.info(f"Python HTTP trigger function `{FUNCTION_ROUTE}` received a request.")
    
    output_model = FunctionResponseModel(
        success=False,
        required_confidence_score=MIN_CONFIDENCE_SCORE
    )
    
    try:
        error_text = "An error occurred during processing."
        error_code = 422

        func_timer = MeasureRunTime()
        func_timer.start()

        # Validate request
        mime_type = req.headers.get("Content-Type")
        if mime_type not in VALID_DI_PREBUILT_READ_LAYOUT_MIME_TYPES:
            return func.HttpResponse(
                f"This function only supports: {', '.join(VALID_DI_PREBUILT_READ_LAYOUT_MIME_TYPES)}. Got: {mime_type}",
                status_code=error_code
            )

        req_body = req.get_body()
        if not req_body:
            return func.HttpResponse(
                "Please provide a PDF document in the request body.",
                status_code=error_code
            )

        # Process document
        doc_page_imgs = load_visual_obj_bytes_to_pil_imgs_dict(
            req_body, mime_type, starting_idx=1, pdf_img_dpi=100
        )

        # Extract text using Document Intelligence
        with MeasureRunTime() as di_timer:
            poller = di_client.begin_analyze_document(
                model_id="prebuilt-read",
                analyze_request=AnalyzeDocumentRequest(bytes_source=req_body),
            )
            di_result = poller.result()
            output_model.di_raw_response = di_result.as_dict()
            
            processed_content_docs = doc_intel_result_processor.process_analyze_result(
                analyze_result=di_result,
                doc_page_imgs=doc_page_imgs,
                on_error="raise",
            )
            merged_processed_content_docs = (
                doc_intel_result_processor.merge_adjacent_text_content_docs(
                    processed_content_docs
                )
            )

        di_result_docs: list[Document] = processed_content_docs
        output_model.di_extracted_text = "\n".join(
            doc.content for doc in di_result_docs if doc.content is not None
        )
        output_model.di_time_taken_secs = di_timer.time_taken

        # Prepare LLM messages
        content_openai_message = convert_processed_di_docs_to_openai_message(
            merged_processed_content_docs, role="user"
        )
        input_messages = [
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            content_openai_message,
        ]
        output_model.llm_input_messages = input_messages

        # Get LLM response
        with MeasureRunTime() as llm_timer:
            llm_result = aoai_client.chat.completions.create(
                messages=input_messages,
                model=AOAI_LLM_DEPLOYMENT,
                response_format={"type": "json_object"},
            )
        output_model.llm_time_taken_secs = llm_timer.time_taken

        # Process LLM response
        output_model.llm_reply_message = llm_result.choices[0].to_dict()
        output_model.llm_raw_response = llm_result.choices[0].message.content
        llm_structured_response = LLMExtractedFieldsModel(
            **json.loads(llm_result.choices[0].message.content)
        )

        # Add confidence scores
        result = {}
        min_field_confidence_score = 1
        is_any_field_missing = False

        for field, value in llm_structured_response.__dict__.items():
            # Find matches in Document Intelligence content
            matches = find_matching_di_words(
                value, di_result,
                match_func=lambda v, c: v.lower() == c.lower()
            )
            
            if not matches:
                matches = find_matching_di_words(
                    value, di_result,
                    match_func=lambda v, c: jellyfish.levenshtein_distance(v.lower(), c.lower()) <= 1
                )
            
            if not matches:
                matches = find_matching_di_lines(
                    value, di_result,
                    match_func=lambda v, c: v.lower() in c.lower()
                )

            # Calculate confidence score
            field_confidence_score = merge_confidence_scores(
                scores=[match.confidence for match in matches],
                no_values_replacement=0.0,
                multiple_values_replacement_func=min,
            )

            # Store result
            result[field] = FieldWithConfidenceModel(
                value=value,
                doc_intel_content_matches_count=len(matches),
                confidence=field_confidence_score,
                normalized_polygons=[m.normalized_polygon for m in matches]
            )

            # Update confidence metrics
            if not value:
                is_any_field_missing = True
            if field_confidence_score < min_field_confidence_score:
                min_field_confidence_score = field_confidence_score

        # Finalize results
        output_model.result = ExtractedFieldsWithConfidenceModel(**result)
        output_model.min_extracted_field_confidence_score = min_field_confidence_score
        output_model.requires_human_review = (
            is_any_field_missing or 
            min_field_confidence_score < MIN_CONFIDENCE_SCORE
        )

        # Draw bounding boxes
        pil_img = doc_page_imgs[1]
        for field_name, field_value in output_model.result.__dict__.items():
            for polygon in field_value.normalized_polygons:
                pixel_based_polygon = scale_flat_poly_list(
                    polygon,
                    existing_scale=(1, 1),
                    new_scale=(pil_img.width, pil_img.height),
                )
                pixel_based_polygon_dict = flat_poly_list_to_poly_dict_list(
                    pixel_based_polygon
                )
                pil_img = draw_polygon_on_pil_img(
                    pil_img=pil_img,
                    polygon=pixel_based_polygon_dict,
                    outline_color="blue",
                    outline_width=3,
                )

        pil_img = resize_img_by_max(pil_img, max_height=1000, max_width=1000)
        output_model.result_img_with_bboxes = pil_img_to_base64_bytes(pil_img)

        # Return successful response
        output_model.success = True
        output_model.func_time_taken_secs = func_timer.stop()
        return func.HttpResponse(
            body=output_model.model_dump_json(),
            mimetype="application/json",
            status_code=200,
        )

    except Exception as e:
        # Return error response
        output_model.success = False
        output_model.error_text = f"{error_text} Error: {str(e)}"
        output_model.func_time_taken_secs = func_timer.stop()
        logging.exception(output_model.error_text)
        return func.HttpResponse(
            body=output_model.model_dump_json(),
            mimetype="application/json",
            status_code=error_code,
        )