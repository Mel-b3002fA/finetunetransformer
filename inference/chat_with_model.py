import sys
import os
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, filename='joi_chat.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Add parent directory to path for model import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
logging.info("Python path: %s", sys.path)

# Import from model.transformer
try:
    from model.transformer import GPT, GPTConfig
    logging.info("Successfully imported GPT, GPTConfig from model.transformer")
except ImportError as e:
    logging.error("Failed to import from model.transformer: %s", str(e))
    print(f"Error: Could not import from model/transformer.py. Ensure model/transformer.py exists and is accessible. Error: {str(e)}")
    raise

# Device setup
device = 'mps' if torch.backends.mps.is_available() else 'cpu'  # Prefer MPS on MacBook, fallback to CPU
model_path = 'out/finetuned_distilgpt2'
block_size = 128  # Matches tokenizer max_length in train.py

# Load model and tokenizer
try:
    config = GPTConfig()
    model = GPT.from_pretrained("distilgpt2").to(device)
    model.load_state_dict(torch.load(os.path.join(model_path, 'pytorch_model.bin'), map_location=device), strict=False)
    model.eval()
    tokenizer = model.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logging.info("Model and tokenizer loaded successfully from %s", model_path)
except Exception as e:
    logging.error("Failed to load model or tokenizer from %s: %s", model_path, str(e))
    raise FileNotFoundError(f"Failed to load model or tokenizer from {model_path}: {str(e)}")

# Chat loop with history and parameter tuning
history = []
debug_mode = False
gen_params = {
    'temperature': 0.7,  # Adjusted for Joi-like responses
    'top_k': 50,
    'top_p': 0.9
}

print("Commands: 'exit/quit/bye' to quit, 'set param=value' to tune (e.g., 'set temp=0.7 k=50 p=0.9'), 'debug on/off' to toggle debug, 'history' to show past exchanges")
try:
    while True:
        prompt = input("You: ").strip()

        # Handle commands
        if prompt.lower() in ['exit', 'quit', 'bye']:
            print("Goodbye!")
            logging.info("Chat session terminated by user")
            break
        elif prompt.lower() == 'history':
            print("\n--- Chat History ---")
            for i, (p, r) in enumerate(history[-5:], 1):
                print(f"{i}. You: {p}")
                print(f"   Joi: {r}")
            continue
        elif prompt.lower().startswith('debug'):
            debug_mode = 'on' in prompt.lower()
            print(f"Debug mode {'enabled' if debug_mode else 'disabled'}")
            logging.info("Debug mode set to %s", debug_mode)
            continue
        elif prompt.lower().startswith('set'):
            parts = prompt.split()
            for part in parts[1:]:
                if '=' in part:
                    key, value = part.split('=')
                    try:
                        value = float(value)
                        if key in ['temp', 'temperature']:
                            gen_params['temperature'] = value
                        elif key in ['k', 'top_k']:
                            gen_params['top_k'] = int(value)
                        elif key in ['p', 'top_p']:
                            gen_params['top_p'] = value
                        logging.info("Set parameter %s=%s", key, value)
                    except ValueError:
                        print(f"Invalid value for {key}: {value}")
                        logging.warning("Invalid parameter value: %s=%s", key, value)
            print(f"Current parameters: {gen_params}")
            continue

        # Build context with history
        context = ""
        for past_prompt, past_response in history[-5:]:  # Limit to last 5 exchanges
            context += f"Human: {past_prompt} || Assistant: {past_response} "
        context += f"Human: {prompt} || Assistant: "
        input_ids = tokenizer(context, return_tensors='pt', truncation=True, max_length=block_size).input_ids.to(device)

        # Generate response
        try:
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=100,
                    temperature=gen_params['temperature'],
                    top_k=gen_params['top_k'],
                    top_p=gen_params['top_p']
                )
            response = tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True)
            print(f"Joi: {response}")
            logging.info("Generated response for prompt '%s': %s", prompt, response)

            # Store in history
            history.append((prompt, response))

            # Debug output
            if debug_mode:
                print(f"Input IDs shape: {input_ids.shape}")
                print(f"Output IDs shape: {output_ids.shape}")
                logging.debug("Input IDs shape: %s, Output IDs shape: %s", input_ids.shape, output_ids.shape)
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            logging.error("Generation error: %s", str(e))

except KeyboardInterrupt:
    print("\n👋 Goodbye!")
    logging.info("Chat session interrupted by user")