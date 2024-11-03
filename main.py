import os
import configparser
import pdfplumber
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from docx import Document

# 설정 파일 읽기
config = configparser.ConfigParser()
config.read('config.ini')

# 리포트 폴더 경로
report_folder = config['Paths']['report_folder']

# PDF 파일에서 텍스트 추출
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# Word 파일에서 텍스트 추출
def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

# 리포트 폴더에서 파일 읽기
report_texts = []
labels = []  # 레이블을 필요에 따라 설정

for filename in os.listdir(report_folder):
    if filename.endswith('.pdf'):
        pdf_path = os.path.join(report_folder, filename)
        report_texts.append(extract_text_from_pdf(pdf_path))
        labels.append('your_label')  # 레이블 설정
    elif filename.endswith('.docx'):
        docx_path = os.path.join(report_folder, filename)
        report_texts.append(extract_text_from_docx(docx_path))
        labels.append('your_label')  # 레이블 설정

# 데이터프레임으로 변환
data = pd.DataFrame({'report_text': report_texts, 'label': labels})

# 특징과 레이블 분리
X = data['report_text']  # 보고서 텍스트
y = data['label']  # 레이블 (예: 긍정/부정)

# TF-IDF 벡터화
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# 모델 선택 및 학습
model = LogisticRegression()
model.fit(X_train, y_train)

# 예측 및 평가
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'모델 정확도: {accuracy:.2f}')
