import json
import os
import random
import re
from openai import OpenAI

from sentences_list import english_sentences_list, korean_translations_list

# 환경 변수에서 API 키 가져오기
api_key = os.getenv('OPENAI_API_KEY')

# OpenAI 클라이언트 인스턴스 생성
client = OpenAI(api_key=api_key)

def generate_sentences(topic, length, fluency_score, accuracy_score, reference, vulnerable_pronunciation):
    # ChatGPT를 사용하여 영어 문장과 한글 번역을 생성
    completion = client.chat.completions.create(
        #model="gpt-3.5-turbo-16k",
        #model="gpt-3.5-turbo",
        # model="gpt-4o",
        model="gpt-4o-mini",
        # model="o1-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant who recommends 5 English sentences that can help improve the "
                    "English pronunciation of Korean speakers. You must reflect the given topic and ensure each "
                    "sentence does not exceed the given word length. Based on the user's fluency/accuracy scores "
                    "between 1 and 5, if the score is higher than 4, you will recommend random sentences that "
                    "reflect the given topic and length. If the score is lower than 4, you will recommend sentences "
                    "that share similar pronunciation/patterns with the reference or strongly incorporate the "
                    "vulnerable phoneme. If 'reference' is 'None' but 'vulnerable_pronunciation' is not None, ignore "
                    "the reference but still include the vulnerable phoneme. If both are 'None', recommend new and "
                    "diverse sentences that satisfy the theme and word count conditions."
                )
            },
            {
                "role": "user",
                "content": (
                    "Generate 5 pairs of English sentences with their Korean translations. Each pair should be in "
                    "the format: 'English sentence - Korean translation'. (Without any explanation, separator, index, "
                    "or ordering. Just return five 'value-value' pairs.)"
                )
            },
            {
                "role": "user",
                "content": (
                    "Response must not include details like ordering: '1. 2.', or explaining: 'english-sentence', etc. "
                    "I'll show you an example of the return format: "
                    "'Watching sunsets in Bali is mesmerizing.-발리의 일몰 감상은 매혹적이에요.',"
                    "'Jogging in the park refreshes me.-공원에서 조깅하는 건 저를 기분 좋게 해줘요.',"
                    "'Pottery making is truly therapeutic.-도예 제작은 정말 치유적이에요.',"
                    "'Origami fascinates me endlessly.-접기에 끊임없는 매력을 느껴요.',"
                    "'Skiing in the Swiss Alps was thrilling.-스위스 알프스에서 스키를 탔어요.'"
                )
            },
            {
                "role": "user",
                "content": (
                    f"The theme of the recommended sentences should be related to '{topic}'. "
                    f"Each English sentence must not exceed '{length}' words. Additionally, at least one instance "
                    f"of the CMU phoneme '{vulnerable_pronunciation}' should appear if it is not 'None'."
                )
            },
            {
                "role": "user",
                "content": (
                    f"The previous user received a fluency score of '{fluency_score}' and an accuracy score of "
                    f"'{accuracy_score}'. Based on these scores, you should generate sentences that "
                    f"appropriately refer to '{reference}' (unless it is 'None') and include the phoneme "
                    f"'{vulnerable_pronunciation}' if it is not 'None'. If '{reference}' or '{vulnerable_pronunciation}' "
                    f"is 'None', ignore them. But if only '{reference}' is 'None', then ignore the reference but still "
                    f"incorporate '{vulnerable_pronunciation}' as much as possible. If both are 'None', generate "
                    f"new and diverse sentences that satisfy the theme and word count."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Especially, if the user's scores are not good (below 4), please make an effort to ensure that "
                    f"the CMU phoneme '{vulnerable_pronunciation}' is reflected as much as possible."
                )
            }
        ]
    )

    # ChatGPT 응답을 우선 줄바꿈 기준으로 분할
    sentences = completion.choices[0].message.content.strip().split("\n")

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 예외처리 추가 부분 <<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # 만약 한 줄만 존재한다면(즉 len(sentences) == 1),
    # 안에 실제로 여러 문장이 콤마(,)로 이어져 있을 수 있으므로 처리
    if len(sentences) == 1:
        line = sentences[0].strip()

        # 1) 대괄호 [ ... ] 로 감싸여 있다면 벗겨냄
        if line.startswith("[") and line.endswith("]"):
            line = line[1:-1].strip()

        # 2) 콤마(,)로 여러 문장이 이어져 있는지 확인
        #    예: 'Going to gym feels good.-헬스장...,Yoga gives me energy.-...'
        if line.count(" - ") > 1 or line.count(".-") > 1 or line.count(",") > 1:
            # 콤마를 기준으로 먼저 분할
            splitted = line.split(",")
            # 공백/인용부호 제거하고 유효한 문장만 다시 sentences에 담음
            splitted = [x.strip().strip('"').strip() for x in splitted if x.strip()]
            if len(splitted) > 1:
                sentences = splitted

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 예외처리 추가 부분 끝 <<<<<<<<<<<<<<<<<<<<<<<<<<<<

    english_sentences = []
    korean_translations = []

    # (영어 - 한국어) 분할을 위한 헬퍼 함수
    def parse_sentence_pair(s: str):
        # 먼저 기존 규칙인 " - " 로 나눌 수 있는지 시도
        if " - " in s:
            parts = s.split(" - ", 1)
            eng = parts[0].strip()
            kor = parts[1].strip()
            return eng, kor
        # 없다면 ".-" 로 나누는 예외처리 (ex: "Going to gym feels good.-헬스장...")
        elif ".-" in s:
            parts = s.split(".-", 1)
            eng = (parts[0].strip() + ".").strip()
            kor = parts[1].strip()
            return eng, kor
        else:
            return None, None

    # 각 문장에 대해 영어/한국어 파트 분류
    for sentence in sentences:
        clean_sentence = sentence.strip().strip('"').strip()
        if not clean_sentence:
            continue
        eng, kor = parse_sentence_pair(clean_sentence)
        if eng and kor:
            english_sentences.append(eng)
            korean_translations.append(kor)

    return english_sentences, korean_translations, sentences


def lambda_handler(event, context):
    headers = {
        "Access-Control-Allow-Origin": "*",  # 모든 출처 허용
        "Access-Control-Allow-Methods": "*",  # 모든 메서드 허용
        "Access-Control-Allow-Headers": '*',  # 모든 헤더 허용
    }
    
    # Body 데이터를 JSON으로 파싱
    body = event.get('body', '{}')
    body_params = json.loads(body)

    # 파라미터 추출
    topic = body_params.get('topic', 'daily life')
    length = body_params.get('length', '10')
    fluency_score = body_params.get('score1', '3.0')
    accuracy_score = body_params.get('score2', '3.0')
    reference = body_params.get('reference', 'None')
    vulnerable_pronunciation = body_params.get('vulnerable', 'None')
    
    # ChatGPT로 문장 생성
    english_sentences, korean_translations, sentences = generate_sentences(
        topic, length, fluency_score, accuracy_score, reference, vulnerable_pronunciation
    )
    is_list = False

    print("Generated English Sentences:", english_sentences)
    print("Generated Korean Translations:", korean_translations)
    print("출력 결과", sentences)
    print("참조문장", reference)

    # 영어 문장은 존재하지만 한국어 번역이 None인 경우 해당 pair 무시
    english_sentences = [
        sen for sen, trans in zip(english_sentences, korean_translations)
        if trans and trans.lower() != 'none'
    ]
    korean_translations = [
        trans for trans in korean_translations
        if trans and trans.lower() != 'none'
    ]

    # 생성된 문장이 5개 미만일 경우 리스트에서 보충
    if len(english_sentences) < 5:
        additional_count = 5 - len(english_sentences)
        is_list = True
        
        # 리스트에서 추가 문장을 선택 (영어와 한국어가 둘 다 존재하는 경우에만 추가)
        available_additional_count = 0
        additional_indices = []
        while available_additional_count < additional_count:
            index = random.randint(0, len(english_sentences_list) - 1)
            if english_sentences_list[index] and korean_translations_list[index]:
                additional_indices.append(index)
                available_additional_count += 1

        # 선택된 인덱스를 사용해 영어 및 한국어 문장 추가
        english_sentences += [english_sentences_list[i] for i in additional_indices]
        korean_translations += [korean_translations_list[i] for i in additional_indices]

    # 영어 문장 리스트에서 임의로 5개 선택
    selected_indices = random.sample(range(len(english_sentences)), 5)
    selected_sentences = [english_sentences[i] for i in selected_indices]
    selected_translations = [korean_translations[i] for i in selected_indices]

    return {
        "statusCode": 200,
        "headers": {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type"
        },
        "body": json.dumps({
            "is_list": is_list,
            "sentences": {
                "sen1": selected_sentences[0],
                "sen2": selected_sentences[1],
                "sen3": selected_sentences[2],
                "sen4": selected_sentences[3],
                "sen5": selected_sentences[4]
            },
            "translations": {
                "sen_trans_1": selected_translations[0],
                "sen_trans_2": selected_translations[1],
                "sen_trans_3": selected_translations[2],
                "sen_trans_4": selected_translations[3],
                "sen_trans_5": selected_translations[4]
            }
        }, ensure_ascii=False)
    }
