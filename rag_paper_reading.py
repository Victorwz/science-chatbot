import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
import numpy as np
import json
import torch.nn.functional as F
from openai import OpenAI
import faiss
import voyageai


# Load existing embeddings and papers
paper_embeddings = np.load("paper_embeddings/embeddings_final.npy")

dimension = paper_embeddings.shape[1]  # Get embedding dimension

# Initialize FAISS index
index = faiss.IndexFlatIP(dimension)  # Use Inner Product similarity
index.add(paper_embeddings)

with open("paper_embeddings/paper_final.json", "r") as f:
    papers = json.load(f)

# Initialize models and tokenizer
llama_model_name = "OpenScholar/Llama-3.1_OpenScholar-8B"

# Initialize Llama model
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
llama_model = AutoModelForCausalLM.from_pretrained(
    llama_model_name,
    torch_dtype=torch.float16,
    device_map="cuda"
)

client = OpenAI()

def generate_embedding(text):
    # Generate embedding using OpenAI's API
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
        encoding_format="float"
    )
    return np.array(response.data[0].embedding)

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
            return np.array(response.embeddings[0])
            
        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt
                print(f"Failed after {max_retries} attempts: {str(e)}")
                return None
            print(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff

def find_relevant_papers(query_embedding, top_k=5):
    # Perform similarity search using FAISS
    scores, indices = index.search(query_embedding.reshape(1, -1), top_k)
    
    # Get the papers and their similarity scores
    relevant_papers = [papers[idx] for idx in indices[0]]
    similarities = scores[0]  # Scores are already cosine similarities
    
    return relevant_papers, similarities


def format_papers_context(papers, similarities):
    relevant_papers = []
    context = "Based on the following relevant papers:\n\n"
    for paper, similarity in zip(papers, similarities):
        original_paper = json.load(open(paper['path']))
        relevant_papers.append(original_paper)
        context += f"Title: {original_paper['title']}\n"
        context += f"Published Time: {original_paper['published_time']}"
        context += f"Abstract: {original_paper['abstract']}\n"
        # context += f"Similarity Score: {similarity:.3f}\n\n"
    return context, relevant_papers


# Generate response using Llama model
def generate_response(message):
    try:
        # Generate embedding for the query
        query_embedding = get_voyage_embedding(message)
        
        # Find relevant papers
        relevant_papers, similarities = find_relevant_papers(query_embedding)

        # Create context from relevant papers
        context, relevant_papers = format_papers_context(relevant_papers, similarities)
        
        # Prepare the prompt with context
        full_prompt = f"{context}\n\nQuestion: {message}\nAnswer:"
        
        # Generate response using Llama model
        inputs = llama_tokenizer(full_prompt, return_tensors="pt").to(llama_model.device)
        
        with torch.no_grad():
            outputs = llama_model.generate(
                inputs.input_ids,
                max_new_tokens=500,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=llama_tokenizer.eos_token_id
            )
        
        response = llama_tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

        # Return all values as a tuple in the same order as the outputs list
        return (
            response,
            relevant_papers[0]['title'],
            relevant_papers[0]['abstract'],
            f"Similarity Score: {similarities[0]:.3f}",
            relevant_papers[1]['title'],
            relevant_papers[1]['abstract'],
            f"Similarity Score: {similarities[1]:.3f}",
            relevant_papers[2]['title'],
            relevant_papers[2]['abstract'],
            f"Similarity Score: {similarities[2]:.3f}"
        )
    
    except Exception as e:
        error_msg = f"An error occurred: {str(e)}\nPlease check your OpenAI API key and make sure all dependencies are properly installed."
        # Return empty strings for all fields in case of error
        return (error_msg, "", "", "", "", "", "", "", "", "")

# Create Gradio interface with multiple outputs
with gr.Blocks(title="RAG-Enhanced Llama-3.1 Chat Interface") as iface:
    gr.Markdown("# RAG-Enhanced Llama-3.1 Chat Interface")
    gr.Markdown("Ask questions about papers in the database. The system will find relevant papers and use them to generate informed responses.")
    
    with gr.Row():
        input_text = gr.Textbox(
            lines=3,
            placeholder="Enter your question...",
            label="Question"
        )
    
    with gr.Row():
        submit_btn = gr.Button("Submit")
    
    with gr.Row():
        response_box = gr.Textbox(
            label="Generated Response",
            lines=10
        )
    
    # Create three columns for the papers
    with gr.Row():
        with gr.Column():
            paper1_title = gr.Textbox(label="Paper 1 Title")
            paper1_similarity = gr.Textbox(label="Similarity Score")
            paper1_abstract = gr.Textbox(label="Abstract", lines=8)
            
        with gr.Column():
            paper2_title = gr.Textbox(label="Paper 2 Title")
            paper2_similarity = gr.Textbox(label="Similarity Score")
            paper2_abstract = gr.Textbox(label="Abstract", lines=8)
            
        with gr.Column():
            paper3_title = gr.Textbox(label="Paper 3 Title")
            paper3_similarity = gr.Textbox(label="Similarity Score")
            paper3_abstract = gr.Textbox(label="Abstract", lines=8)
    
    submit_btn.click(
        fn=generate_response,
        inputs=input_text,
        outputs=[
            response_box,
            paper1_title, paper1_abstract, paper1_similarity,
            paper2_title, paper2_abstract, paper2_similarity,
            paper3_title, paper3_abstract, paper3_similarity
        ]
    )
    
    # Add example questions
    gr.Examples(
        examples=[
            ["Can you recommend some papers about perovskite which are published in last 5 years?"],
            ["Explain the concept of attention mechanisms in deep learning."],
            ["What are the main challenges in few-shot learning?"]
        ],
        inputs=input_text
    )

# Launch the interface
if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0",
        server_port=None,
        share=True)
    # generate_response("Can you recommend some papers about perovskite which are published in last 5 years?")