import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import PyPDF2
import pdfplumber
from PIL import Image
import numpy as np
import pandas as pd
from loguru import logger
import openai
import base64
from io import BytesIO

@dataclass
class DocumentChunk:
    """Document chunk structure"""
    chunk_id: str
    content: str
    page_number: int
    chunk_index: int
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

@dataclass
class TableData:
    """Structured table data"""
    table_id: str
    headers: List[str]
    rows: List[List[Any]]
    metadata: Dict[str, Any]
    json_representation: Dict[str, Any]

@dataclass
class ChartData:
    """Structured chart data"""
    chart_id: str
    chart_type: str
    title: str
    x_axis: Dict[str, Any]
    y_axis: Dict[str, Any]
    data_series: List[Dict[str, Any]]
    insights: str
    metadata: Dict[str, Any]

class PDFProcessor:
    """PDF document processor"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.chunk_size = config.get("retrieval", {}).get("chunk_size", 512)
        self.chunk_overlap = config.get("retrieval", {}).get("chunk_overlap", 50)
    
    async def process_pdf(self, file_path: str) -> Dict[str, Any]:
        """Process PDF file and extract all content"""
        
        logger.info(f"Processing PDF: {file_path}")
        
        result = {
            "file_path": file_path,
            "text_chunks": [],
            "tables": [],
            "charts": [],
            "metadata": {}
        }
        
        try:
            # Extract text content
            text_content = await self._extract_text(file_path)
            
            # Extract tables
            tables = await self._extract_tables(file_path)
            
            # Extract images/charts
            charts = await self._extract_charts(file_path)
            
            # Create text chunks
            chunks = self._create_chunks(text_content)
            
            result["text_chunks"] = chunks
            result["tables"] = tables
            result["charts"] = charts
            result["metadata"] = self._extract_metadata(file_path)
            
            logger.info(f"PDF processing completed: {len(chunks)} chunks, {len(tables)} tables, {len(charts)} charts")
            
        except Exception as e:
            logger.error(f"PDF processing failed: {str(e)}")
            raise
        
        return result
    
    async def _extract_text(self, file_path: str) -> str:
        """Extract text content from PDF"""
        
        text = ""
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
        
        except Exception as e:
            logger.error(f"Text extraction failed: {str(e)}")
        
        return text
    
    async def _extract_tables(self, file_path: str) -> List[TableData]:
        """Extract tables from PDF"""
        
        tables = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()
                    
                    for table_idx, table in enumerate(page_tables):
                        if table and len(table) > 1:
                            table_data = self._process_table(
                                table,
                                page_num + 1,
                                table_idx
                            )
                            tables.append(table_data)
        
        except Exception as e:
            logger.error(f"Table extraction failed: {str(e)}")
        
        return tables
    
    def _process_table(
        self,
        table: List[List[Any]],
        page_number: int,
        table_index: int
    ) -> TableData:
        """Process extracted table into structured format"""
        
        # Clean table data
        cleaned_table = []
        for row in table:
            cleaned_row = [str(cell).strip() if cell else "" for cell in row]
            if any(cleaned_row):  # Skip empty rows
                cleaned_table.append(cleaned_row)
        
        if not cleaned_table:
            return None
        
        # Extract headers (first row)
        headers = cleaned_table[0]
        rows = cleaned_table[1:]
        
        # Create JSON representation
        json_data = []
        for row in rows:
            row_dict = {}
            for i, header in enumerate(headers):
                if i < len(row):
                    row_dict[header] = row[i]
            json_data.append(row_dict)
        
        table_id = f"table_p{page_number}_t{table_index}"
        
        return TableData(
            table_id=table_id,
            headers=headers,
            rows=rows,
            metadata={
                "page_number": page_number,
                "table_index": table_index,
                "row_count": len(rows),
                "column_count": len(headers)
            },
            json_representation={"data": json_data}
        )
    
    async def _extract_charts(self, file_path: str) -> List[ChartData]:
        """Extract and analyze charts from PDF"""
        
        charts = []
        
        try:
            # Extract images from PDF
            images = self._extract_images_from_pdf(file_path)
            
            # Analyze each image
            for img_idx, (image, page_num) in enumerate(images):
                chart_data = await self._analyze_chart_image(
                    image,
                    page_num,
                    img_idx
                )
                if chart_data:
                    charts.append(chart_data)
        
        except Exception as e:
            logger.error(f"Chart extraction failed: {str(e)}")
        
        return charts
    
    def _extract_images_from_pdf(self, file_path: str) -> List[Tuple[Image.Image, int]]:
        """Extract images from PDF pages"""
        
        images = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Convert page to image
                    page_image = page.to_image(resolution=150)
                    pil_image = page_image.original
                    images.append((pil_image, page_num + 1))
        
        except Exception as e:
            logger.error(f"Image extraction failed: {str(e)}")
        
        return images
    
    async def _analyze_chart_image(
        self,
        image: Image.Image,
        page_number: int,
        chart_index: int
    ) -> Optional[ChartData]:
        """Analyze chart image using vision model"""
        
        try:
            # Convert image to base64
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Prepare prompt for chart analysis
            prompt = """
            Analyze this financial chart and extract structured data.
            Return a JSON object with:
            {
                "chart_type": "line|bar|pie|scatter|other",
                "title": "chart title",
                "x_axis": {"name": "axis name", "values": [], "unit": "unit"},
                "y_axis": {"name": "axis name", "values": [], "unit": "unit"},
                "data_series": [
                    {"name": "series name", "data_points": [{"x": value, "y": value}]}
                ],
                "insights": "key insights from the chart"
            }
            """
            
            # Call vision model (GPT-4V or similar)
            # Note: This is a placeholder - actual implementation would use real API
            chart_analysis = await self._call_vision_model(img_base64, prompt)
            
            if chart_analysis:
                chart_data = json.loads(chart_analysis)
                
                return ChartData(
                    chart_id=f"chart_p{page_number}_c{chart_index}",
                    chart_type=chart_data.get("chart_type", "unknown"),
                    title=chart_data.get("title", ""),
                    x_axis=chart_data.get("x_axis", {}),
                    y_axis=chart_data.get("y_axis", {}),
                    data_series=chart_data.get("data_series", []),
                    insights=chart_data.get("insights", ""),
                    metadata={
                        "page_number": page_number,
                        "chart_index": chart_index
                    }
                )
        
        except Exception as e:
            logger.error(f"Chart analysis failed: {str(e)}")
        
        return None
    
    async def _call_vision_model(self, img_base64: str, prompt: str) -> str:
        """Call vision model for image analysis"""
        
        # Placeholder for vision model call
        # In production, this would call GPT-4V or similar
        return json.dumps({
            "chart_type": "line",
            "title": "Financial Performance",
            "x_axis": {"name": "Year", "values": [2021, 2022, 2023], "unit": "year"},
            "y_axis": {"name": "Revenue", "values": [100, 150, 200], "unit": "million"},
            "data_series": [
                {
                    "name": "Revenue",
                    "data_points": [
                        {"x": 2021, "y": 100},
                        {"x": 2022, "y": 150},
                        {"x": 2023, "y": 200}
                    ]
                }
            ],
            "insights": "Revenue shows consistent growth over three years"
        })
    
    def _create_chunks(self, text: str) -> List[DocumentChunk]:
        """Create text chunks with overlap"""
        
        chunks = []
        
        # Split text into sentences
        sentences = text.replace('\n', ' ').split('. ')
        
        current_chunk = ""
        chunk_index = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if adding this sentence exceeds chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size:
                if current_chunk:
                    # Create chunk
                    chunk = DocumentChunk(
                        chunk_id=f"chunk_{chunk_index}",
                        content=current_chunk,
                        page_number=0,  # Would need to track actual page
                        chunk_index=chunk_index,
                        metadata={"length": len(current_chunk)}
                    )
                    chunks.append(chunk)
                    
                    # Keep overlap for next chunk
                    overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else ""
                    current_chunk = overlap_text + " " + sentence
                    chunk_index += 1
                else:
                    current_chunk = sentence
            else:
                current_chunk += ". " + sentence if current_chunk else sentence
        
        # Add last chunk
        if current_chunk:
            chunk = DocumentChunk(
                chunk_id=f"chunk_{chunk_index}",
                content=current_chunk,
                page_number=0,
                chunk_index=chunk_index,
                metadata={"length": len(current_chunk)}
            )
            chunks.append(chunk)
        
        return chunks
    
    def _extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract document metadata"""
        
        metadata = {
            "file_name": os.path.basename(file_path),
            "file_size": os.path.getsize(file_path),
            "file_path": file_path
        }
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                metadata["page_count"] = len(pdf_reader.pages)
                
                # Extract PDF metadata
                if pdf_reader.metadata:
                    metadata["title"] = pdf_reader.metadata.get('/Title', '')
                    metadata["author"] = pdf_reader.metadata.get('/Author', '')
                    metadata["subject"] = pdf_reader.metadata.get('/Subject', '')
                    metadata["creator"] = pdf_reader.metadata.get('/Creator', '')
        
        except Exception as e:
            logger.error(f"Metadata extraction failed: {str(e)}")
        
        return metadata

class TableProcessor:
    """Advanced table processing and semantic understanding"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.financial_metrics = self._load_financial_metrics()
    
    def _load_financial_metrics(self) -> Dict[str, Any]:
        """Load predefined financial metrics"""
        
        return {
            "revenue": ["营业收入", "收入", "销售额", "revenue", "sales"],
            "profit": ["利润", "净利润", "profit", "net income"],
            "cost": ["成本", "费用", "cost", "expense"],
            "rd_expense": ["研发费用", "研发投入", "R&D", "research"],
            "assets": ["资产", "总资产", "assets", "total assets"],
            "liabilities": ["负债", "总负债", "liabilities"],
            "equity": ["股东权益", "净资产", "equity", "shareholders equity"]
        }
    
    async def process_table(self, table_data: TableData) -> Dict[str, Any]:
        """Process table for semantic understanding"""
        
        # Identify table type
        table_type = self._identify_table_type(table_data)
        
        # Extract financial metrics
        metrics = self._extract_financial_metrics(table_data)
        
        # Identify time periods
        periods = self._identify_time_periods(table_data)
        
        # Calculate relationships
        relationships = self._calculate_relationships(metrics)
        
        return {
            "table_id": table_data.table_id,
            "table_type": table_type,
            "metrics": metrics,
            "periods": periods,
            "relationships": relationships,
            "original_data": table_data.json_representation
        }
    
    def _identify_table_type(self, table_data: TableData) -> str:
        """Identify the type of financial table"""
        
        headers_text = " ".join(table_data.headers).lower()
        
        if any(term in headers_text for term in ["income", "profit", "revenue", "损益", "利润"]):
            return "income_statement"
        elif any(term in headers_text for term in ["balance", "assets", "liabilities", "资产负债"]):
            return "balance_sheet"
        elif any(term in headers_text for term in ["cash", "flow", "现金流"]):
            return "cash_flow"
        else:
            return "general"
    
    def _extract_financial_metrics(self, table_data: TableData) -> Dict[str, Any]:
        """Extract financial metrics from table"""
        
        metrics = {}
        
        for row in table_data.rows:
            if not row:
                continue
            
            # Check first column for metric names
            metric_name = str(row[0]).lower() if row else ""
            
            # Match against known metrics
            for metric_key, aliases in self.financial_metrics.items():
                if any(alias in metric_name for alias in aliases):
                    # Extract values from row
                    values = []
                    for i in range(1, len(row)):
                        try:
                            # Try to parse as number
                            value = self._parse_number(row[i])
                            if value is not None:
                                values.append(value)
                        except:
                            pass
                    
                    if values:
                        metrics[metric_key] = values
                    break
        
        return metrics
    
    def _parse_number(self, value: Any) -> Optional[float]:
        """Parse number from various formats"""
        
        if value is None or value == "":
            return None
        
        # Convert to string and clean
        str_value = str(value).replace(",", "").replace("，", "")
        str_value = str_value.replace("￥", "").replace("$", "").replace("%", "")
        
        try:
            return float(str_value)
        except:
            return None
    
    def _identify_time_periods(self, table_data: TableData) -> List[str]:
        """Identify time periods in table headers"""
        
        periods = []
        
        for header in table_data.headers:
            header_str = str(header)
            
            # Check for year patterns (e.g., 2023, 2023年)
            import re
            year_pattern = r'20\d{2}'
            years = re.findall(year_pattern, header_str)
            
            if years:
                periods.extend(years)
            
            # Check for quarter patterns (e.g., Q1, Q2, 1季度)
            quarter_pattern = r'Q[1-4]|[1-4]季度'
            quarters = re.findall(quarter_pattern, header_str)
            
            if quarters:
                periods.extend(quarters)
        
        return periods
    
    def _calculate_relationships(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Calculate relationships between metrics"""
        
        relationships = []
        
        # Check for profit = revenue - cost
        if "revenue" in metrics and "cost" in metrics and "profit" in metrics:
            relationships.append({
                "type": "calculation",
                "formula": "profit = revenue - cost",
                "verified": self._verify_calculation(
                    metrics["profit"],
                    metrics["revenue"],
                    metrics["cost"],
                    operation="subtract"
                )
            })
        
        # Check for ROE = profit / equity
        if "profit" in metrics and "equity" in metrics:
            relationships.append({
                "type": "ratio",
                "formula": "ROE = profit / equity",
                "name": "Return on Equity"
            })
        
        return relationships
    
    def _verify_calculation(
        self,
        result: List[float],
        operand1: List[float],
        operand2: List[float],
        operation: str
    ) -> bool:
        """Verify mathematical relationship"""
        
        try:
            if len(result) != len(operand1) or len(result) != len(operand2):
                return False
            
            for i in range(len(result)):
                if operation == "subtract":
                    calculated = operand1[i] - operand2[i]
                elif operation == "add":
                    calculated = operand1[i] + operand2[i]
                else:
                    return False
                
                # Allow small difference for floating point
                if abs(calculated - result[i]) > 0.01:
                    return False
            
            return True
        
        except:
            return False

class MultiModalPreprocessor:
    """Main multi-modal document preprocessor"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pdf_processor = PDFProcessor(config)
        self.table_processor = TableProcessor(config)
    
    async def preprocess_document(self, file_path: str) -> Dict[str, Any]:
        """Preprocess document with multi-modal extraction"""
        
        logger.info(f"Starting multi-modal preprocessing: {file_path}")
        
        # Determine file type
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == ".pdf":
            # Process PDF
            pdf_result = await self.pdf_processor.process_pdf(file_path)
            
            # Enhanced table processing
            processed_tables = []
            for table in pdf_result["tables"]:
                processed_table = await self.table_processor.process_table(table)
                processed_tables.append(processed_table)
            
            pdf_result["processed_tables"] = processed_tables
            
            return pdf_result
        
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    async def batch_preprocess(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Batch preprocess multiple documents"""
        
        tasks = []
        for file_path in file_paths:
            task = asyncio.create_task(self.preprocess_document(file_path))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        logger.info(f"Batch preprocessing completed: {len(results)} documents")
        
        return results