import json

def count_json_elements_from_file(filename):
    try:
        # 파일에서 JSON 데이터 로드
        with open(filename, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # JSON이 리스트 형태인지 확인
        if isinstance(data, list):
            return len(data)
        else:
            return "JSON data is not a list."
    except (json.JSONDecodeError, FileNotFoundError) as e:
        return f"Error: {str(e)}"

# 로컬 파일에서 데이터 읽기
filename = "시니컬_학습_데이터.json"
count = count_json_elements_from_file(filename)
print("Element count:", count)
