# interface.py
import streamlit as st
import requests

st.title("📰 Fake News Detector from URL")

url = st.text_input("Enter a news article URL:")

if st.button("Check Article"):
    if url:
        response = requests.post("http://127.0.0.1:5000/predict", json={"url": url})
        if response.status_code == 200:
            # if response valid, api return the result
            data = response.json()
            st.success(f"### Prediction: {data['prediction']}")
            st.write(f"**REAL:** {data['confidence']['REAL']:.3f}")
            st.write(f"**FAKE:** {data['confidence']['FAKE']:.3f}")
            st.markdown("---")
            st.write("📝 **Article snippet:**")
            st.text(data["article_text_snippet"])
        else:
            try:
                # for case where the input is null or invalid text and not link
                error_data = response.json()
                st.error(
                    f"Error {response.status_code}: {error_data.get('error', 'Unknown error')}"
                )
            except Exception:
                # for case when link provided is invalid
                st.error(f"Error {response.status_code}: {response.text}")

    else:
        st.warning("Please enter a URL before checking.")
