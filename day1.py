import os
from openai import OpenAI

# OpenAI 클라이언트 생성
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")  # API 키 설정
)

# 시스템 메시지 생성 함수
def create_system_message(content):
    return {"role": "system", "content": content}

# 사용자 메시지 생성 함수
def create_user_message(content):
    return {"role": "user", "content": content}

# 메시지 전송 및 응답 처리 함수
def send_chat_request(system_message, user_message, model="gpt-4o-mini"):
    messages = [
        create_system_message(system_message),
        create_user_message(user_message)
    ]
    return client.chat.completions.create(
        model=model,
        messages=messages)
    

# 시스템 및 사용자 메시지 설정
# 문제1
# user_msg = "Solve the following problem step-by-step: 23 + 47"
# system_msg = "덧셈"
# "1. 문제의 이해"
# "2. 숫자의 자리수 분리"
# "3. 각 자리수를 더하기"
# "4. 각 자리수 더한것을 합치기"
# "5. 결과 검증"

# 문제2
# user_msg = "Solve the following problem step-by-step: 123 - 58"
# system_msg = "뺄셈"
# "1. 문제의 이해"
# "2. 숫자의 자리수 분리"
# "3. 일의 자리부터 뺄셈 진행, -인 경우 윗자리에서 값을 빌려옴"
# "4. 각 자리수 뺀것을 합치기"
# "5. 결과검증"

# 문제3
user_msg = "Solve the following problem step-by-step: 345 + 678 - 123"
system_msg = "뺄셈"
"1. 문제의 이해"
"2. 숫자의 자리수 분리"
"3. 일의 자리부터 뺄셈 진행, -인 경우 윗자리에서 값을 빌려옴"
"4. 각 자리수 뺀것을 합치기"
"5. 결과검증"
"덧셈"
"1. 문제의 이해"
"2. 숫자의 자리수 분리"
"3. 각 자리수를 더하기"
"4. 각 자리수 더한것을 합치기"
"5. 결과 검증"

# OpenAI API 호출
completion = send_chat_request(system_msg, user_msg)

# 결과 출력
print(completion.choices[0].message)
