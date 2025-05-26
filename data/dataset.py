import os
import json
import logging
import re
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, filename='dataset_preparation.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

def load_whatsapp_chats(file_path: str) -> List[Dict[str, str]]:
    """Load and parse WhatsApp chat logs from _chat.txt."""
    if not os.path.exists(file_path):
        logging.error("WhatsApp chat file %s not found", file_path)
        raise FileNotFoundError(f"WhatsApp chat file {file_path} not found")
    
    conversations = []
    current_user = None
    current_message = []
    
    # Regex to match WhatsApp export format: [date, time] username: message
    pattern = re.compile(r'^\[\d{2}/\d{2}/\d{4}, \d{2}:\d{2}:\d{2}\] (.*?): (.*)$')
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Check if line matches WhatsApp format
            match = pattern.match(line)
            if not match:
                logging.warning("Skipping malformed line: %s", line)
                continue
            
            username, message = match.groups()
            message = message.strip()
            if not message:  # Skip empty messages
                continue
            
            # Map usernames to User or Joi
            if username == '</>':
                if current_user and current_message:
                    prompt = "Human: " + " ".join(current_message) + " || Assistant: "
                    if prompt.strip() != "Human:  || Assistant: ":
                        conversations.append({
                            "prompt": prompt,
                            "response": ""
                        })
                current_user = "User"
                current_message = [message]
            elif username == 'Tilman':
                if current_user == "User" and current_message:
                    if message:  # Only add non-empty responses
                        conversations[-1]["response"] = message
                current_user = "Joi"
                current_message = []
            elif current_user:
                current_message.append(message)
    
    # Append the last conversation
    if current_user == "User" and current_message:
        prompt = "Human: " + " ".join(current_message) + " || Assistant: "
        if prompt.strip() != "Human:  || Assistant: ":
            conversations.append({
                "prompt": prompt,
                "response": ""
            })
    
    logging.info("Loaded %d conversation pairs from %s", len(conversations), file_path)
    return conversations

def load_joi_data(file_path: str) -> List[Dict[str, str]]:
    """Load curated Joi data from joi.jsonl."""
    if not os.path.exists(file_path):
        logging.warning("Joi data file %s not found, skipping", file_path)
        return []
    
    conversations = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                prompt = f"Human: {data.get('prompt', '')} || Assistant: "
                response = data.get('response', '')
                if prompt.strip() != "Human:  || Assistant: " and response.strip():
                    conversations.append({
                        "prompt": prompt,
                        "response": response
                    })
            except json.JSONDecodeError as e:
                logging.warning("Skipping invalid JSON line in %s: %s", file_path, str(e))
    
    logging.info("Loaded %d curated Joi pairs from %s", len(conversations), file_path)
    return conversations

def save_processed_dataset(conversations: List[Dict[str, str]], output_path: str):
    """Save conversations to processed_dataset.jsonl."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    valid_conversations = [
        conv for conv in conversations
        if conv["prompt"].strip() != "Human:  || Assistant: " and conv["response"].strip()
    ]
    with open(output_path, 'w', encoding='utf-8') as f:
        for conv in valid_conversations:
            f.write(json.dumps({"prompt": conv["prompt"] + conv["response"]}) + '\n')
    logging.info("Saved %d conversations to %s", len(valid_conversations), output_path)
    return len(valid_conversations)

def main():
    # File paths
    whatsapp_path = "data/_chat.txt"
    joi_data_path = "data/joi.jsonl"
    output_path = "data/processed_dataset.jsonl"
    
    # Load data
    whatsapp_conversations = load_whatsapp_chats(whatsapp_path)
    joi_conversations = load_joi_data(joi_data_path)
    
    # Combine and save
    all_conversations = whatsapp_conversations + joi_conversations
    num_saved = save_processed_dataset(all_conversations, output_path)
    print(f"Processed dataset saved to {output_path} with {num_saved} conversation pairs")

if __name__ == "__main__":
    main()