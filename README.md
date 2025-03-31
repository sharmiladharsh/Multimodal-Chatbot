🤖 Multimodal Q&A Chatbot

A powerful AI chatbot that answers questions from documents (PDF, DOCX, CSV), images, and videos using cutting-edge NLP and computer vision models.

Built with Streamlit, Hugging Face Transformers, BLIP, and FAISS.

📌 Features

✅ 🔍 Semantic search using SentenceTransformers + FAISS✅ 📄 Text QA from PDF, DOCX, CSV using BERT✅ 📊 Handles structured data (tables in DOCX and CSV)✅ 🖼️ Visual Q&A using BLIP (Salesforce)✅ 🎞️ Video QA by extracting frames using OpenCV✅ 🧠 User-friendly Streamlit UI

📂 Supported File Types

Type

Formats

Documents

.pdf, .docx, .csv

Images

.jpg, .jpeg, .png

Videos

.mp4, .avi, .mov

🧠 Models Used

Task

Model Name

Sentence Embedding

all-MiniLM-L6-v2 (SentenceTransformers)

Text Q&A

bert-large-uncased-whole-word-masking-finetuned-squad

Visual Q&A

Salesforce/blip-vqa-base

🚀 Installation

1️⃣ Clone the repository

git clone https://github.com/yourusername/multimodal-chatbot.git
cd multimodal-chatbot

2️⃣ Create a virtual environment (optional but recommended)

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3️⃣ Install dependencies

pip install -r requirements.txt

4️⃣ Run the application

streamlit run app.py

👨‍💻 Author

Made by Sharmila Devi A S
