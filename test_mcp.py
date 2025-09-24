import os
import json
import streamlit as st
from dotenv import load_dotenv
from tempfile import TemporaryDirectory
from git import Repo
import zipfile
import shutil
import boto3
import mimetypes
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_aws.llms import BedrockLLM
from langchain_core.prompts import ChatPromptTemplate

# Load AWS keys
load_dotenv()

# Initialize Bedrock clients
# bedrock_runtime_us = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
# bedrock_runtime_in = boto3.client(service_name='bedrock-runtime', region_name='ap-south-1')

# Initialize LLM model
def get_llm():
    return BedrockLLM(
        client=bedrock_runtime_in,
        region_name="ap-south-1",
        model_id='meta.llama3-8b-instruct-v1:0'
    )

# File handling functions
def is_text_file(file_path):
    """Determine if a file is a text file that should be processed."""
    # Get MIME type
    mime_type, _ = mimetypes.guess_type(file_path)
    
    # Skip binary and media files
    if mime_type and (mime_type.startswith(('image/', 'audio/', 'video/', 'application/pdf', 
                                          'application/vnd.ms-', 'application/vnd.openxmlformats'))):
        return False
    
    # Common code and config file extensions to include
    code_extensions = {
        '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.c', '.cpp', '.h', '.cs', '.go', '.rs',
        '.php', '.rb', '.swift', '.kt', '.sh', '.bash', '.ps1', '.html', '.css', '.scss',
        '.md', '.txt', '.json', '.yml', '.yaml', '.xml', '.toml', '.ini', '.cfg', '.conf',
        '.Dockerfile', 'Dockerfile', '.dockerignore', '.gitignore', '.env.example',
        'Jenkinsfile', '.groovy', '.tf', '.hcl'
    }
    
    # Check file extension
    ext = Path(file_path).suffix.lower()
    filename = Path(file_path).name
    
    if ext in code_extensions or filename in code_extensions:
        return True
    
    # For files without extension or with uncommon extensions, try to read a few bytes
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            f.read(1024)  # Try to read first 1KB
        return True  # If we can read it as text, consider it a text file
    except (UnicodeDecodeError, IOError):
        return False  # Not a text file or can't be read

def read_file_content(file_path):
    """Read the content of a file safely."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

def chunk_text(text, chunk_size=4000, chunk_overlap=200):
    """Split text into chunks if it's too large."""
    if len(text) <= chunk_size:
        return [text]
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

# File processing
def process_files(repo_path, output_dir):
    """Process all code files in the repository."""
    file_summaries = {}
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Walk through all files in the repository
    for root, _, files in os.walk(repo_path):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, repo_path)
            
            # Skip hidden files and directories
            if any(part.startswith('.') for part in rel_path.split(os.sep) if part not in ['.gitignore', '.dockerignore']):
                continue
                
            # Check if it's a processable text file
            if not is_text_file(file_path):
                continue
                
            # Read file content
            content = read_file_content(file_path)
            if content.startswith("Error reading file:"):
                continue
                
            # Generate a clean filename for the JSON output
            clean_filename = rel_path.replace(os.sep, '_').replace('.', '_')
            json_file = os.path.join(output_dir, f"{clean_filename}_summary.json")
            
            # Process the file content
            file_info = analyze_file_content(content, rel_path)
            
            # Save file summary
            with open(json_file, 'w') as f:
                json.dump(file_info, f, indent=2)
                
            # Add to overall summaries
            file_summaries[rel_path] = file_info
    
    # Save the overall file summaries index
    with open(os.path.join(output_dir, "file_summaries_index.json"), 'w') as f:
        json.dump(file_summaries, f, indent=2)
        
    return file_summaries

def analyze_file_content(content, file_path):
    """Analyze file content and extract relevant information."""
    llm = get_llm()
    
    # Check if content needs chunking
    chunks = chunk_text(content)
    file_info = {
        "path": file_path,
        "size_bytes": len(content),
        "chunks": len(chunks),
        "summary": "",
        "functions": [],
        "classes": [],
        "api_endpoints": []
    }
    
    # Process each chunk and combine results
    all_results = []
    
    for i, chunk in enumerate(chunks):
        try:
            # Create an analysis prompt for the LLM
            prompt = ChatPromptTemplate.from_template("""
            Analyze the following code/file content and provide:
            1. A brief summary (2-3 sentences max) of what this file/code does
            2. List of functions/methods defined (name, purpose)
            3. List of classes defined (name, purpose)
            4. Any API endpoints defined (route, method, purpose)
            
            Content from file '{file_path}' {chunk_info}:
            ```
            {content}
            ```
            
            Format your response as follows:
            SUMMARY: <brief summary of this file>
            FUNCTIONS: 
            - function_name: brief purpose
            - another_function: what it does
            
            CLASSES: 
            - class_name: brief purpose
            - another_class: what it does
            
            API_ENDPOINTS: 
            - /path (GET): brief purpose
            - /another/path (POST): what it does
            
            Note: Only include items that are actually defined in this code. If there are none in a category, just write "None found" for that section.
            """)
            
            chunk_info = f"(chunk {i+1} of {len(chunks)})" if len(chunks) > 1 else ""
            
            response = llm.invoke(prompt.format(
                file_path=file_path,
                content=chunk,
                chunk_info=chunk_info
            ))
            
            # Parse the response
            parsed_result = parse_llm_response(response)
            all_results.append(parsed_result)
            
        except Exception as e:
            # Handle errors gracefully
            print(f"Error processing chunk {i+1} of file {file_path}: {str(e)}")
            # Add minimal info for this chunk
            all_results.append({
                "summary": f"Error analyzing chunk {i+1}",
                "functions": [],
                "classes": [],
                "api_endpoints": []
            })
    
    # Combine results from all chunks
    if all_results:
        # Use the first valid chunk's summary as the main summary
        for result in all_results:
            if result.get("summary") and not result.get("summary").startswith("Error"):
                file_info["summary"] = result.get("summary", "")
                break
        
        # If no valid summary was found
        if not file_info["summary"]:
            file_info["summary"] = f"Could not analyze file {file_path}"
        
        # Safely combine functions, classes, and API endpoints
        for result in all_results:
            # Make sure we're dealing with lists
            functions = result.get("functions", [])
            if isinstance(functions, list):
                file_info["functions"].extend(functions)
                
            classes = result.get("classes", [])
            if isinstance(classes, list):
                file_info["classes"].extend(classes)
                
            endpoints = result.get("api_endpoints", [])
            if isinstance(endpoints, list):
                file_info["api_endpoints"].extend(endpoints)
    
    return file_info

def parse_llm_response(response):
    """Parse the LLM response to extract structured information."""
    result = {
        "summary": "",
        "functions": [],
        "classes": [],
        "api_endpoints": []
    }
    
    # Basic parsing of the response
    lines = response.split('\n')
    current_section = None
    section_content = []
    
    for line in lines:
        line = line.strip()
        
        if line.startswith("SUMMARY:"):
            if current_section and section_content:
                # Process the previous section
                process_section_content(result, current_section, section_content)
                section_content = []
            
            current_section = "summary"
            result["summary"] = line[len("SUMMARY:"):].strip()
            
        elif line.startswith("FUNCTIONS:"):
            if current_section and section_content:
                # Process the previous section
                process_section_content(result, current_section, section_content)
                section_content = []
                
            current_section = "functions"
            functions_str = line[len("FUNCTIONS:"):].strip()
            
            # Try to parse as JSON directly
            if try_parse_json(functions_str, result, "functions"):
                current_section = None  # Reset if successful
            else:
                # Will collect content for manual parsing
                section_content = []
                
        elif line.startswith("CLASSES:"):
            if current_section and section_content:
                # Process the previous section
                process_section_content(result, current_section, section_content)
                section_content = []
                
            current_section = "classes"
            classes_str = line[len("CLASSES:"):].strip()
            
            # Try to parse as JSON directly
            if try_parse_json(classes_str, result, "classes"):
                current_section = None  # Reset if successful
            else:
                # Will collect content for manual parsing
                section_content = []
                
        elif line.startswith("API_ENDPOINTS:"):
            if current_section and section_content:
                # Process the previous section
                process_section_content(result, current_section, section_content)
                section_content = []
                
            current_section = "api_endpoints"
            api_str = line[len("API_ENDPOINTS:"):].strip()
            
            # Try to parse as JSON directly
            if try_parse_json(api_str, result, "api_endpoints"):
                current_section = None  # Reset if successful
            else:
                # Will collect content for manual parsing
                section_content = []
        
        elif current_section:
            # Collect content for the current section
            if line:  # Skip empty lines
                section_content.append(line)
    
    # Process the last section if needed
    if current_section and section_content:
        process_section_content(result, current_section, section_content)
    
    return result

def try_parse_json(json_str, result, section_key):
    """Try to parse a string as JSON and store in result dict."""
    try:
        if json_str and json_str.strip():
            if json_str.startswith("[") and json_str.endswith("]"):
                parsed_data = json.loads(json_str)
                
                # Validate the format - ensure each item has the necessary keys
                if section_key == "functions" or section_key == "classes":
                    validated_items = []
                    for item in parsed_data:
                        if isinstance(item, dict):
                            # Ensure required keys exist with defaults if missing
                            validated_item = {
                                "name": item.get("name", "Unnamed"),
                                "purpose": item.get("purpose", "No description")
                            }
                            validated_items.append(validated_item)
                    result[section_key] = validated_items
                elif section_key == "api_endpoints":
                    validated_items = []
                    for item in parsed_data:
                        if isinstance(item, dict):
                            validated_item = {
                                "route": item.get("route", "/unknown"),
                                "method": item.get("method", "GET"),
                                "purpose": item.get("purpose", "No description")
                            }
                            validated_items.append(validated_item)
                    result[section_key] = validated_items
                return True
    except json.JSONDecodeError:
        pass
    return False

def process_section_content(result, section_key, content_lines):
    """Process collected content for a section."""
    if not content_lines:
        return
        
    if section_key == "summary":
        # Join all lines for summary
        result["summary"] = " ".join(content_lines)
    
    elif section_key in ["functions", "classes"]:
        # Parse each line as a function or class description
        for line in content_lines:
            if line.startswith("-") or line.startswith("*"):
                line = line[1:].strip()
                
            # Try to extract name and purpose
            if ":" in line:
                name, purpose = line.split(":", 1)
                item = {"name": name.strip(), "purpose": purpose.strip()}
            else:
                item = {"name": line.strip(), "purpose": "No description"}
            
            result[section_key].append(item)
    
    elif section_key == "api_endpoints":
        # Parse each line as an API endpoint
        for line in content_lines:
            if line.startswith("-") or line.startswith("*"):
                line = line[1:].strip()
                
            # Try various formats for API endpoints
            if " - " in line:
                route_method, purpose = line.split(" - ", 1)
                
                if "(" in route_method and ")" in route_method:
                    # Format: /path (METHOD) - purpose
                    route, method = route_method.split("(", 1)
                    method = method.split(")", 1)[0]
                    item = {"route": route.strip(), "method": method.strip(), "purpose": purpose.strip()}
                else:
                    # Format: /path - purpose (assume GET)
                    item = {"route": route_method.strip(), "method": "GET", "purpose": purpose.strip()}
            else:
                # Just the route
                item = {"route": line.strip(), "method": "GET", "purpose": "No description"}
            
            result[section_key].append(item)

# Generate comprehensive summaries
def generate_comprehensive_summary(file_summaries):
    """Generate comprehensive summaries based on file summaries."""
    llm = get_llm()
    
    # Convert file summaries to a summarized format
    summary_content = []
    for path, info in file_summaries.items():
        summary_content.append(f"File: {path}")
        summary_content.append(f"Summary: {info['summary']}")
        
        if info['functions']:
            summary_content.append("Functions:")
            for func in info['functions'][:5]:  # Limit to prevent prompt size issues
                summary_content.append(f"- {func.get('name', 'Unnamed')}: {func.get('purpose', 'No description')}")
        
        if info['classes']:
            summary_content.append("Classes:")
            for cls in info['classes'][:5]:  # Limit to prevent prompt size issues
                summary_content.append(f"- {cls.get('name', 'Unnamed')}: {cls.get('purpose', 'No description')}")
        
        if info['api_endpoints']:
            summary_content.append("API Endpoints:")
            for api in info['api_endpoints'][:5]:  # Limit to prevent prompt size issues
                summary_content.append(f"- {api.get('route', '/unknown')} ({api.get('method', 'METHOD')}): {api.get('purpose', 'No description')}")
        
        summary_content.append("")  # Add blank line between files
    
    summary_text = "\n".join(summary_content)
    
    # Generate different types of summaries
    summaries = {}
    
    # High Level Summary
    high_level_prompt = ChatPromptTemplate.from_template("""
    Based on the following file summaries from a code repository, provide a high-level summary of the repository 
    including its purpose, features, and general architecture. Limit your response to 5-7 sentences.
    
    File Summaries:
    {summary_text}
    """)
    
    summaries["high_level"] = llm.invoke(high_level_prompt.format(summary_text=summary_text))
    
    # Technical LLD
    technical_prompt = ChatPromptTemplate.from_template("""
    Based on the following file summaries from a code repository, provide a low-level technical design (LLD) of the repository.
    Include details like key classes, functions, their purposes, and how they interact. Organize your response by major components.
    
    File Summaries:
    {summary_text}
    """)
    
    summaries["technical_lld"] = llm.invoke(technical_prompt.format(summary_text=summary_text))
    
    # Technical Workflow
    workflow_prompt = ChatPromptTemplate.from_template("""
    Based on the following file summaries from a code repository, describe the technical workflow of the repository.
    Explain how the different modules interact, any setup processes, and the typical usage flow.
    
    File Summaries:
    {summary_text}
    """)
    
    summaries["workflow"] = llm.invoke(workflow_prompt.format(summary_text=summary_text))
    
    # API Documentation
    api_prompt = ChatPromptTemplate.from_template("""
    Based on the following file summaries from a code repository, extract and describe all API-related details.
    Include routes, methods, expected input/output, and authentication if present. Organize by endpoint category if possible.
    
    File Summaries:
    {summary_text}
    """)
    
    summaries["api_docs"] = llm.invoke(api_prompt.format(summary_text=summary_text))
    
    return summaries

# --- Main Processing Logic ---
def process_repo(repo_input_method, summary_type, git_url=None, zip_file=None, local_path=None):
    try:
        with TemporaryDirectory() as temp_dir:
            repo_path = temp_dir
            summaries_dir = os.path.join(temp_dir, "file_summaries")
            
            # Create summaries directory
            os.makedirs(summaries_dir, exist_ok=True)

            # Get repository content
            try:
                if repo_input_method == "GitHub URL":
                    Repo.clone_from(git_url, to_path=temp_dir)
                elif repo_input_method == "Upload ZIP":
                    zip_path = os.path.join(temp_dir, "uploaded_repo.zip")
                    with open(zip_path, "wb") as f:
                        f.write(zip_file.getbuffer())
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                elif repo_input_method == "Local Path":
                    shutil.copytree(local_path, temp_dir, dirs_exist_ok=True)
            except Exception as e:
                st.error(f"Error obtaining repository content: {str(e)}")
                return f"Failed to process repository: {str(e)}"

            # Process all files and generate summaries
            with st.spinner('Processing files...'):
                try:
                    file_summaries = process_files(repo_path, summaries_dir)
                    if not file_summaries:
                        return "No processable files found in the repository."
                except Exception as e:
                    st.error(f"Error processing files: {str(e)}")
                    return f"Failed to process files: {str(e)}"
            
            # Generate comprehensive summaries
            with st.spinner('Generating comprehensive summary...'):
                try:
                    summaries = generate_comprehensive_summary(file_summaries)
                except Exception as e:
                    st.error(f"Error generating summary: {str(e)}")
                    return f"Failed to generate summary: {str(e)}"
            
            # Return the requested summary type
            summary_mapping = {
                "High Level Summary": "high_level",
                "Technical LLD": "technical_lld",
                "Technical Workflow and Documentation": "workflow",
                "API Documentations": "api_docs"
            }
            
            selected_summary = summaries.get(summary_mapping.get(summary_type, "high_level"))
            if not selected_summary:
                return "Summary generation failed. Please try a different summary type."
                
            return selected_summary
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return f"An unexpected error occurred: {str(e)}"

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="Repo Summary Generator", layout='wide')
    st.title("GitHub/Repo Summary Generator")

    # Inject consistent styling
    st.markdown("""
        <style>
        .markdown-text-container {
            font-size: 16px;
            line-height: 1.6;
        }
        </style>
    """, unsafe_allow_html=True)

    input_method = st.sidebar.selectbox("Select Input Method", ("GitHub URL", "Upload ZIP", "Local Path"))
    summary_type = st.sidebar.selectbox("Select Summary Type", [
        "High Level Summary",
        "Technical LLD",
        "Technical Workflow and Documentation",
        "API Documentations"
    ])

    git_url = None
    zip_file = None
    local_path = None

    if input_method == "GitHub URL":
        git_url = st.sidebar.text_input("Enter GitHub Repository URL")
    elif input_method == "Upload ZIP":
        zip_file = st.sidebar.file_uploader("Upload ZIP file", type=["zip"])
    elif input_method == "Local Path":
        local_path = st.sidebar.text_input("Enter Local Path")

    if st.sidebar.button("Generate Summary"):
        if (input_method == "GitHub URL" and git_url) or \
           (input_method == "Upload ZIP" and zip_file) or \
           (input_method == "Local Path" and local_path):
            with st.spinner('Processing...'):
                summary = process_repo(input_method, summary_type, git_url, zip_file, local_path)
            st.subheader(f"{summary_type}")
            st.markdown(summary, unsafe_allow_html=False)
        else:
            st.error("Please provide the required input.")

if __name__ == "__main__":
    main()
