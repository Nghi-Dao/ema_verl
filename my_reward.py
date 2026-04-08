from math_verify import parse, verify

def extract_boxed_answer(text):
    # Find the last occurrence of \boxed{
    start_idx = text.rfind("\\boxed{")
    if start_idx == -1:
        return None
        
    content_start = start_idx + 7
    brace_count = 1
    
    # Count braces until we find the matching closing brace
    for i in range(content_start, len(text)):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            
        if brace_count == 0:
            return text[content_start:i].strip()
            
    return None



def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    try:
        extracted_answer = extract_boxed_answer(solution_str)

        if extracted_answer is None:
            return 0.0

        pred_parsed = parse(extracted_answer, parsing_timeout=None)
        gt_parsed = parse(str(ground_truth), parsing_timeout=None)

        is_correct = verify(pred_parsed, gt_parsed)
        return 1.0 if is_correct else 0.0

    except Exception as e:
        print(f"REWARD CRASH: {e}")
        return 0.0

