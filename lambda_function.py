import json
import urllib.request
import urllib.parse
import logging
import time

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    """
    Lambda handler for text summarization using third-party Inference API.
    Supports multiple models and robust error handling.
    """
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization',
        'Access-Control-Allow-Methods': 'POST, OPTIONS'
    }
    
    # Handle preflight requests
    if event.get('httpMethod') == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': headers,
            'body': ''
        }
    
    try:
        # Parse request body
        body = json.loads(event.get('body', '{}'))
        text = body.get('text', '').strip()
        summary_type = body.get('summaryType', 'bullet_points')
        length = body.get('length', 'medium')
        
        # Validate input
        if not text:
            return {
                'statusCode': 400,
                'headers': headers,
                'body': json.dumps({'error': 'Text is required'})
            }
        
        if len(text) < 50:
            return {
                'statusCode': 400,
                'headers': headers,
                'body': json.dumps({'error': 'Text too short. Please provide at least 50 characters.'})
            }
        
        # Truncate text to meet model constraints
        max_length = 5000
        if len(text) > max_length:
            text = text[:max_length] + "..."
            logger.info("Text truncated to 5000 characters.")
        
        # Generate summary using external summarization API
        summary = call_summarizer_api(text, length)
        
        # Format summary
        if summary_type == 'bullet_points':
            summary = format_as_bullet_points(summary)
        elif summary_type == 'numbered_list':
            summary = format_as_numbered_list(summary)

        # Compute compression stats
        original_words = len(text.split())
        summary_words = len(summary.split())
        compression_ratio = round((1 - summary_words / original_words) * 100) if original_words > 0 else 0
        
        return {
            'statusCode': 200,
            'headers': headers,
            'body': json.dumps({
                'summary': summary,
                'originalWords': original_words,
                'summaryWords': summary_words,
                'compressionRatio': compression_ratio,
                'processingTime': context.get_remaining_time_in_millis()
            })
        }
        
    except json.JSONDecodeError:
        return {
            'statusCode': 400,
            'headers': headers,
            'body': json.dumps({'error': 'Invalid JSON in request body'})
        }
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({'error': f'Internal server error: {str(e)}'})
        }


def call_summarizer_api(text, length):
    """
    Attempt summarization using a tiered fallback of pre-trained models via API.
    """
    # Placeholder models (replace with actual API routes or secured endpoints)
    models = [
        "model_A",
        "model_B",
        "model_C"
    ]
    
    length_map = {
        'short': {'max_length': 80, 'min_length': 20},
        'medium': {'max_length': 150, 'min_length': 30},
        'long': {'max_length': 300, 'min_length': 50}
    }
    
    params = length_map.get(length, length_map['medium'])
    
    for model in models:
        try:
            logger.info(f"Trying model: {model}")
            summary = try_model(model, text, params)
            if summary:
                return summary
        except Exception as e:
            logger.warning(f"{model} failed: {str(e)}")
    
    # All models failed — use extractive fallback
    return simple_extractive_summary(text, length)


def try_model(model, text, params):
    """
    Make an HTTP request to a hosted model endpoint.
    This is a placeholder for inference API.
    """
    # Endpoint removed for security
    url = f"https://your-inference-api.com/models/{model}"  # Replace in production
    
    payload = {
        "inputs": text,
        "parameters": {
            "max_length": params['max_length'],
            "min_length": params['min_length'],
            "do_sample": False,
            "truncation": True
        }
    }
    
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data)
    req.add_header('Content-Type', 'application/json')
    
    with urllib.request.urlopen(req, timeout=20) as response:
        result = json.loads(response.read().decode('utf-8'))
        
        if isinstance(result, list) and 'summary_text' in result[0]:
            return result[0]['summary_text']
        elif isinstance(result, list) and 'generated_text' in result[0]:
            return result[0]['generated_text']
        elif isinstance(result, dict) and 'error' in result:
            if 'loading' in result['error'].lower():
                time.sleep(2)
                return try_model(model, text, params)
            else:
                raise Exception(result['error'])
        
        raise Exception("Unexpected response format")


def simple_extractive_summary(text, length):
    """
    Simple extractive fallback summary if all model calls fail.
    """
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    if not sentences:
        return "No content to summarize."
    
    scored_sentences = []
    for i, sentence in enumerate(sentences):
        if len(sentence.split()) < 5:
            continue
        score = 0
        words = sentence.lower().split()
        
        if i == 0: score += 3
        elif i == 1: score += 2
        elif i < len(sentences) * 0.3: score += 1
        
        word_count = len(words)
        if 10 <= word_count <= 25: score += 2
        elif 5 <= word_count <= 35: score += 1
        
        important_keywords = ['important', 'key', 'main', 'significant']
        if any(word in words for word in important_keywords):
            score += 1
        
        scored_sentences.append((score, sentence, i))
    
    scored_sentences.sort(key=lambda x: (-x[0], x[2]))
    if length == "short":
        selected = scored_sentences[:2]
    elif length == "long":
        selected = scored_sentences[:6]
    else:
        selected = scored_sentences[:4]
    
    selected.sort(key=lambda x: x[2])
    return '. '.join([s[1] for s in selected]) + '.'


def format_as_bullet_points(text):
    sentences = [s.strip() for s in text.replace('\n', '. ').split('.') if len(s.strip()) > 10]
    bullets = [f"• {s.rstrip('.,!?')}" for s in sentences[:8]]
    return '\n'.join(bullets) if bullets else f"• {text}"


def format_as_numbered_list(text):
    sentences = [s.strip() for s in text.replace('\n', '. ').split('.') if len(s.strip()) > 10]
    numbered = [f"{i+1}. {s.rstrip('.,!?')}" for i, s in enumerate(sentences[:8])]
    return '\n'.join(numbered) if numbered else f"1. {text}"
