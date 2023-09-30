### Core Concept
외국인에게 서울의 유적지를 알려주는 챗봇

### To run
.env.example에서 example 지우고 .env로 만들기  
.env 파일에서 openai api key만 넣으면 동작함 (pinecone이 아니라 chromadb 사용하므로)
```
!pip install streamlit
# other required libraries should also be installed.
```
```
streamlit run web/app.py
```
