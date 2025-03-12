# Example usage:
import json
from pathlib import Path
import os, sys
from openai import OpenAI
import time
from tqdm import tqdm
import numpy as np
import tiktoken
import voyageai

enc = tiktoken.get_encoding("cl100k_base")

def extract_paper_content(paper_data):
    """
    Extract and concatenate all text content from a paper's data structure.
    
    Args:
        paper_data (dict): Dictionary containing paper data including title, abstract, sections, etc.
    
    Returns:
        str: Concatenated text content from the paper
    """
    text_parts = []
    
    # Add title
    if 'title' in paper_data:
        text_parts.append(f"Title: {paper_data['title']}")

    # Add title
    if 'published_time' in paper_data:
        text_parts.append(f"Published Time: {paper_data['published_time']}")
    
    # Add abstract
    if 'abstract' in paper_data:
        text_parts.append(f"Abstract: {paper_data['abstract']}")
    
    # Add main sections
    if 'sections' in paper_data:
        for section in paper_data['sections']:
            section_title = section.get('section', '')
            section_content = section.get('content', '')
            text_parts.append(f"{section_title}: {section_content}")
    
    # Add references if needed
    if 'references' in paper_data:
        for ref in paper_data['references']:
            ref_title = ref.get('title', '')
            ref_idx = ref.get('idx', '')
            text_parts.append(f"References [{ref_idx}]: " + ref_title)
    
    # Join all parts with newlines
    full_text = '\n\n'.join(text_parts)
    
    return full_text



def process_paper_file(file_content):
    """
    Process paper content from file and extract text.
    
    Args:
        file_content (str): JSON string containing paper data
    
    Returns:
        str: Concatenated text content from the paper
    """
    try:
        # Parse the JSON content
        paper_data = json.loads(file_content)
        
        # Extract the text content
        full_text = extract_paper_content(paper_data)
        
        return full_text
        
    except json.JSONDecodeError:
        return "Error: Invalid JSON format"
    except Exception as e:
        return f"Error processing paper: {str(e)}"

def get_embedding(text, model="text-embedding-3-small", client=None):
    """Get embeddings for text using OpenAI API with rate limiting and retry logic."""
    if client is None:
        client = OpenAI()
        
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                model=model,
                input=text,
                encoding_format="float"
            )
            return response.data[0].embedding
            
        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt
                print(f"Failed after {max_retries} attempts: {str(e)}")
                return None
            print(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff

def get_voyage_embedding(text, model="voyage-3-lite", client=None):
    """Get embeddings for text using OpenAI API with rate limiting and retry logic."""
    if client is None:
        client = voyageai.Client()
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            response = client.embed(
                model=model,
                texts=[text]
            )
            return response.embeddings[0]
            
        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt
                print(f"Failed after {max_retries} attempts: {str(e)}")
                return None
            print(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff

def process_papers_to_embeddings(data_dir, batch_size=100, save_every=1000):
    """
    Process papers in directory to embeddings and save them periodically.
    
    Args:
        data_dir (str): Path to directory containing paper JSON files
        batch_size (int): Number of papers to process before getting embeddings
        save_every (int): How often to save embeddings to disk
    """
    client = OpenAI()
    client = voyageai.Client()
    data_path = Path(data_dir)
    
    # Lists to store embeddings and paper IDs
    embeddings = []
    paper_ids = []
    current_batch_texts = []
    current_batch_ids = []
    
    # Output files
    output_dir = Path("paper_embeddings")
    output_dir.mkdir(exist_ok=True)
    
    def save_current_data(embed_list, id_list, suffix=""):
        if not embed_list:
            return
        np.save(output_dir / f"embeddings{suffix}.npy", np.array(embed_list))
        with open(output_dir / f"paper{suffix}.json", "w") as f:
            f.write(json.dumps(id_list, indent=2))
        # np.save(output_dir / f"paper_ids{suffix}.npy", np.array(id_list))
    
    # Process all paper files
    paper_files = []
    for paper in data_path.iterdir():
        for file in paper.iterdir():
            if "processed_data" in file.stem and "json" in file.suffix:
                paper_files.append(file)
    
    for i, file in enumerate(tqdm(paper_files)):
        try:
            # Load and process paper
            paper_id = file.stem
            full_text = process_paper_file(open(file).read())
            truncated_full_text = enc.decode(enc.encode(full_text)[:8160])
            
            # Add to current batch
            current_batch_texts.append(truncated_full_text)
            current_batch_ids.append({"id": paper_id, "text": full_text, "path": str(file)})
            
            # Process batch if it reaches batch_size
            if len(current_batch_texts) >= batch_size:
                print(f"Processing batch of {len(current_batch_texts)} papers...")
                for text, pid in zip(current_batch_texts, current_batch_ids):
                    embedding = get_voyage_embedding(text, client=client)
                    if embedding is not None:
                        embeddings.append(embedding)
                        paper_ids.append(pid)
                
                # Clear current batch
                current_batch_texts = []
                current_batch_ids = []
            
            # Save periodically
            if (i + 1) % save_every == 0:
                print(f"Saving checkpoint at paper {i+1}...")
                save_current_data(embeddings, paper_ids, f"_checkpoint_{i+1}")
        
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
            continue
    
    # Process any remaining papers in the last batch
    if current_batch_texts:
        print(f"Processing final batch of {len(current_batch_texts)} papers...")
        for text, pid in zip(current_batch_texts, current_batch_ids):
            embedding = get_embedding(text, client=client)
            if embedding is not None:
                embeddings.append(embedding)
                paper_ids.append(pid)
    
    # Save final data
    print("Saving final embeddings...")
    save_current_data(embeddings, paper_ids, "_final")
    
    return embeddings, paper_ids

# Example usage
if __name__ == "__main__":
    data_dir = "/share/edc/home/zekunli/mmnc-data/scraped_articles/Nature Communications/Physical sciences/Materials science/"
    embeddings, paper_ids = process_papers_to_embeddings(data_dir)
    print(f"Processed {len(embeddings)} papers successfully")