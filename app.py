import streamlit as st
import pickle

def main():
    st.title("Input Kalimat")
    st.write("Masukkan kalimat di bawah ini:")
    with open('new_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    # Load the KNN model
    with open('knn_model.pkl', 'rb') as f:
        knn_model = pickle.load(f)

    kalimat = st.text_input("Kalimat:")

    if st.button("Prediksi"):
        # Lakukan proses prediksi di sini
        test_input = vectorizer.transform([kalimat])
        predictions = knn_model.predict(test_input)
        labels = ["negatif", "positif"]
        st.write("""## Sentimen ini bernilai """, labels[predictions[0]])

if __name__ == "__main__":
    main()