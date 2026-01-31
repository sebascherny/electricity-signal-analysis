#!/usr/bin/env python3
"""
FastAPI web server for ElectricitySignalAnalyzer.
Allows users to upload CSV files and configure analysis parameters via web interface.
"""

import os
import tempfile
import zipfile
import shutil
import logging
from datetime import datetime
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from electricity_signal_analyzer import ElectricitySignalAnalyzer

# Configure logging to capture both file and memory logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='web_server.log',
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Electricity Signal Analyzer", version="1.0.0")

# Create static directory for serving files
os.makedirs("static", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_upload_form():
    """Serve the main upload form."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Electricity Signal Analyzer</title>
        <link rel="icon" type="image/svg+xml" href="/static/favicon.svg">
        <link rel="icon" type="image/x-icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%2280%22 font-size=%2280%22>🔌</text></svg>">
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                text-align: center;
                margin-bottom: 30px;
            }
            .form-group {
                margin-bottom: 20px;
            }
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
                color: #555;
            }
            input, select, textarea {
                width: 100%;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
                font-size: 14px;
                box-sizing: border-box;
            }
            input[type="file"] {
                padding: 5px;
            }
            .form-row {
                display: flex;
                gap: 15px;
            }
            .form-row .form-group {
                flex: 1;
            }
            button {
                background-color: #007bff;
                color: white;
                padding: 12px 30px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                width: 100%;
                margin-top: 20px;
            }
            button:hover {
                background-color: #0056b3;
            }
            button:disabled {
                background-color: #ccc;
                cursor: not-allowed;
            }
            .help-text {
                font-size: 12px;
                color: #666;
                margin-top: 5px;
            }
            .loading {
                display: none;
                text-align: center;
                margin-top: 20px;
            }
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #3498db;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 2s linear infinite;
                margin: 0 auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔌 Electricity Signal Analyzer</h1>
            <form id="analysisForm" enctype="multipart/form-data">
                <div class="form-group">
                    <label for="csv_file">CSV File:</label>
                    <input type="file" id="csv_file" name="csv_file" accept=".csv" required>
                    <div class="help-text">Upload a CSV file with electricity signal data</div>
                </div>
                
                <div class="form-group">
                    <label for="column_name">Column to Analyze:</label>
                    <input type="text" id="column_name" name="column_name" value="Report_1_IB" required>
                    <div class="help-text">Name of the column containing the signal data</div>
                </div>
                
                <div class="form-row">
                    <div class="form-group">
                        <label for="window_size">Window Size:</label>
                        <input type="number" id="window_size" name="window_size" value="200" min="10" max="1000" required>
                        <div class="help-text">Size of each analysis window</div>
                    </div>
                    <div class="form-group">
                        <label for="step_size">Step Size:</label>
                        <input type="number" id="step_size" name="step_size" value="100" min="1" max="500" required>
                        <div class="help-text">Step size for overlapping windows</div>
                    </div>
                </div>
                
                <div class="form-row">
                    <div class="form-group">
                        <label for="start_index">Start Index (optional):</label>
                        <input type="number" id="start_index" name="start_index" min="0" placeholder="Leave empty for start">
                        <div class="help-text">Starting sample index</div>
                    </div>
                    <div class="form-group">
                        <label for="end_index">End Index (optional):</label>
                        <input type="number" id="end_index" name="end_index" min="1" placeholder="Leave empty for end">
                        <div class="help-text">Ending sample index</div>
                    </div>
                </div>
                
                <div class="form-row">
                    <div class="form-group">
                        <label for="max_windows">Max Windows (optional):</label>
                        <input type="number" id="max_windows" name="max_windows" min="1" max="50" placeholder="Leave empty for all">
                        <div class="help-text">Maximum number of windows to process</div>
                    </div>
                    <div class="form-group">
                        <label for="n_exponents">Number of Exponents:</label>
                        <input type="text" id="n_exponents" name="n_exponents" value="4,5,6,8" required>
                        <div class="help-text">Comma-separated list (e.g., 2,4,6,8)</div>
                    </div>
                </div>
                
                <div class="form-group">
                    <label style="display: flex; width: 40%;">
                        <span>Generate and save plots</span>
                        <input style="width: 30px;" type="checkbox" id="save_plots" name="save_plots" checked>
                    </label>
                    <div class="help-text">Generate visualization plots for each window</div>
                </div>
                
                <button type="submit">🚀 Analyze Signal</button>
            </form>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Processing your signal analysis... This may take a few minutes.</p>
            </div>
        </div>
        
        <script>
            document.getElementById('analysisForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const formData = new FormData();
                const fileInput = document.getElementById('csv_file');
                const file = fileInput.files[0];
                
                if (!file) {
                    alert('Please select a CSV file');
                    return;
                }
                
                // Show loading spinner
                document.getElementById('loading').style.display = 'block';
                document.querySelector('button[type="submit"]').disabled = true;
                
                // Collect form data
                formData.append('csv_file', file);
                formData.append('column_name', document.getElementById('column_name').value);
                formData.append('window_size', document.getElementById('window_size').value);
                formData.append('step_size', document.getElementById('step_size').value);
                formData.append('n_exponents', document.getElementById('n_exponents').value);
                formData.append('save_plots', document.getElementById('save_plots').checked);
                
                // Optional parameters
                const startIndex = document.getElementById('start_index').value;
                const endIndex = document.getElementById('end_index').value;
                const maxWindows = document.getElementById('max_windows').value;
                
                if (startIndex) formData.append('start_index', startIndex);
                if (endIndex) formData.append('end_index', endIndex);
                if (maxWindows) formData.append('max_windows', maxWindows);
                
                try {
                    const response = await fetch('/analyze', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (response.ok) {
                        // Download the ZIP file
                        const blob = await response.blob();
                        const url = window.URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.style.display = 'none';
                        a.href = url;
                        a.download = 'analysis_results.zip';
                        document.body.appendChild(a);
                        a.click();
                        window.URL.revokeObjectURL(url);
                        alert('Analysis completed! Results downloaded as ZIP file.');
                    } else {
                        const error = await response.text();
                        alert('Error: ' + error);
                    }
                } catch (error) {
                    alert('Network error: ' + error.message);
                } finally {
                    // Hide loading spinner
                    document.getElementById('loading').style.display = 'none';
                    document.querySelector('button[type="submit"]').disabled = false;
                }
            });
        </script>
    </body>
    </html>
    """
    return html_content

@app.post("/analyze")
async def analyze_signal(
    csv_file: UploadFile = File(...),
    column_name: str = Form(...),
    window_size: int = Form(...),
    step_size: int = Form(...),
    n_exponents: str = Form(...),
    save_plots: bool = Form(False),
    start_index: Optional[int] = Form(None),
    end_index: Optional[int] = Form(None),
    max_windows: Optional[int] = Form(None)
):
    """Process the uploaded CSV file and return analysis results as ZIP."""
    
    # Create temporary directory for this analysis
    temp_dir = tempfile.mkdtemp(prefix="signal_analysis_")
    log_file_path = None
    
    try:
        logger.info(f"Starting analysis for file: {csv_file.filename}")
        
        # Validate file type
        if not csv_file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV file")
        
        # Parse n_exponents list
        try:
            n_exponents_list = [int(x.strip()) for x in n_exponents.split(',')]
            if not n_exponents_list:
                raise ValueError("Empty list")
        except ValueError:
            raise HTTPException(status_code=400, detail="n_exponents must be comma-separated integers (e.g., 2,4,6)")
        
        # Save uploaded file
        csv_file_path = os.path.join(temp_dir, csv_file.filename)
        with open(csv_file_path, "wb") as buffer:
            content = await csv_file.read()
            buffer.write(content)
        
        # Set up logging to capture analysis logs
        log_file_path = os.path.join(temp_dir, "analysis.log")
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to both root logger and analyzer logger
        root_logger = logging.getLogger()
        analyzer_logger = logging.getLogger('electricity_signal_analyzer')
        root_logger.addHandler(file_handler)
        analyzer_logger.addHandler(file_handler)
        
        try:
            # Initialize analyzer
            analyzer = ElectricitySignalAnalyzer()
            
            # Load the uploaded CSV file
            data = analyzer.load_file(csv_file_path)
            if data is None:
                raise HTTPException(status_code=400, detail="Failed to load CSV file")
            
            # Set column to analyze
            analyzer.choose_column_to_use(column_name)
            
            # Validate column exists
            if column_name not in analyzer.data.columns:
                available_columns = list(analyzer.data.columns)
                raise HTTPException(
                    status_code=400, 
                    detail=f"Column '{column_name}' not found. Available columns: {available_columns}"
                )
            
            # Run analysis
            logger.info(f"Running analysis with parameters: window_size={window_size}, step_size={step_size}, n_exponents={n_exponents_list}")
            
            results = analyzer.iterate_through_windows(
                window_size=window_size,
                step_size=step_size,
                n_exponents_list=n_exponents_list,
                plot_windows=save_plots,
                save_plots=save_plots,
                max_windows=max_windows,
                start_index=start_index,
                end_index=end_index
            )
            
            logger.info(f"Analysis completed. Processed {len(results) if results else 0} windows")
            
        finally:
            # Remove log handlers to avoid memory leaks
            root_logger.removeHandler(file_handler)
            analyzer_logger.removeHandler(file_handler)
            file_handler.close()
        
        # Create ZIP file with results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"signal_analysis_results_{timestamp}.zip"
        zip_path = os.path.join(temp_dir, zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add log file
            if log_file_path and os.path.exists(log_file_path):
                zipf.write(log_file_path, "analysis.log")
            
            # Add generated plots if they exist
            simulation_outputs_dir = "simulation_outputs"
            if save_plots and os.path.exists(simulation_outputs_dir):
                # Find the most recent analysis folder
                analysis_folders = [d for d in os.listdir(simulation_outputs_dir) 
                                  if d.startswith("electricity_analysis_")]
                if analysis_folders:
                    latest_folder = max(analysis_folders)
                    plots_dir = os.path.join(simulation_outputs_dir, latest_folder)
                    
                    if os.path.exists(plots_dir):
                        for root, dirs, files in os.walk(plots_dir):
                            for file in files:
                                if file.endswith('.png'):
                                    file_path = os.path.join(root, file)
                                    arcname = os.path.join("plots", file)
                                    zipf.write(file_path, arcname)
            
            # Add analysis summary
            summary_content = f"""Electricity Signal Analysis Results
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Parameters:
- File: {csv_file.filename}
- Column: {column_name}
- Window Size: {window_size}
- Step Size: {step_size}
- N Exponents: {n_exponents_list}
- Start Index: {start_index if start_index is not None else 'Beginning'}
- End Index: {end_index if end_index is not None else 'End'}
- Max Windows: {max_windows if max_windows is not None else 'All'}
- Plots Generated: {save_plots}

Results:
- Windows Processed: {len(results) if results else 0}
- Cached Analysis Data: {len(analyzer.get_cached_windows_summary())} entries

Files in this ZIP:
- analysis.log: Detailed processing logs
- summary.txt: This summary file
"""
            if save_plots:
                summary_content += "- plots/: Generated visualization plots\n"
            
            zipf.writestr("summary.txt", summary_content)
        
        logger.info(f"Created ZIP file: {zip_path}")
        
        # Return the ZIP file
        async def cleanup_background():
            cleanup_temp_dir(temp_dir)
        
        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename=zip_filename,
            background=cleanup_background
        )
        
    except HTTPException:
        # Clean up temp directory on HTTP errors
        cleanup_temp_dir(temp_dir)
        raise
    except Exception as e:
        # Clean up temp directory on other errors
        cleanup_temp_dir(temp_dir)
        logger.exception(f"Unexpected error during analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

def cleanup_temp_dir(temp_dir: str):
    """Clean up temporary directory."""
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        logger.warning(f"Failed to clean up temporary directory {temp_dir}: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    logger.info("Starting Electricity Signal Analyzer Web Server")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
