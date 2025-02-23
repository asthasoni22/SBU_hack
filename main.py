from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import faiss
import pickle
import json
from langchain.schema import HumanMessage


# Set up API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyDNjC9gL9JKfnXuoo41AlmHTffrQYPO-0Y"

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-pro")


FAISS_INDEX_PATH = "data/faiss_index/index.faiss"
def load_faiss_index(index_path=FAISS_INDEX_PATH):
    """Load FAISS index from a pickle file."""
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index file not found: {index_path}")
    
    if os.path.isdir(index_path):  
        raise IsADirectoryError(f"Expected a file but found a directory: {index_path}")

    try:
        with open(index_path, "rb") as f:
            faiss_index = pickle.load(f)
            
            # ✅ Ensure it's a FAISS object
            if not isinstance(faiss_index, FAISS):
                raise TypeError("Loaded object is not a valid FAISS vector store")
            
            return faiss_index
    except Exception as e:
        print(f"⚠️ Error loading FAISS index: {e}")
        return None



# Retrieve user data from FAISS
def retrieve_user_data(vector_db, user_id, fallback_data_path="data/faiss_index/example_data.json"):
    # If FAISS is available, try retrieving the user data
    if vector_db:
        try:
            results = vector_db.similarity_search(user_id, k=1)  # Fetch closest match
            if results:
                metadata_data = results[0].metadata["data"]

                if isinstance(metadata_data, str):
                    return json.loads(metadata_data)  # Convert JSON string to dict
                elif isinstance(metadata_data, dict):
                    return metadata_data  # Already a dictionary
        except AttributeError:
            print("⚠️ FAISS object is not properly initialized. Using example data.")

    # If FAISS retrieval fails, load example data
    print("⚠️ No match found in FAISS. Loading fallback example data.")
    with open(fallback_data_path, "r", encoding="utf-8") as f:
        return json.load(f)



# Generate explanation for disease prediction
def generate_medical_explanation(llm, user_data):
    features = user_data.get("top_5_features", {})
    disease = user_data.get("disease", "the condition")

    # Categorize features based on positive or negative LIME influence
    positive_features = []
    negative_features = []

    for feature, (value, lime_prob) in features.items():
        explanation = f"- {feature}: Value = {value}, LIME Impact = {lime_prob:.2f}"
        if lime_prob >= 0:
            positive_features.append(explanation)
        else:
            negative_features.append(explanation)

    # Construct the prompt
    prompt = f"""
    The user has been diagnosed with {disease}. Here are the factors influencing this prediction:

    **Contributing Factors:**
    {'\n'.join(positive_features) if positive_features else 'None'}

    **Factors Acting Against the Prediction:**
    {'\n'.join(negative_features) if negative_features else 'None'}

    Explain why these factors impact {disease} in simple terms.
    """

    response = response = llm.invoke(prompt)  # ✅ Pass the string directly
  # FIXED: Correct invocation
    return response.content


# General chat response
def general_chat(llm, user_input):
    response = llm.invoke(HumanMessage(content=user_input))  # FIXED: Correct invocation
    return response.content


def main(user_query, use_fallback=False):
    vector_db = load_faiss_index()  # Load FAISS index
    
    try:
        user_data = retrieve_user_data(vector_db, user_query)
    except Exception as e:
        print(f"⚠️ Error retrieving data from FAISS: {e}")
        user_data = None

    # If no data found, use fallback example data
    if use_fallback or user_data is None:
        print("⚠️ No match found in FAISS. Loading fallback example data.")
        with open("data/faiss_index/example_data.json", "r") as f:
            user_data = json.load(f)  # Load example data
    
    if "top_5_features" in user_data:
        return generate_medical_explanation(llm, user_data)
    else:
        return general_chat(llm, user_query)



if __name__ == "__main__":
    user_query = "12345"  # Use user_id from example_data.json
    print(main(user_query, use_fallback=True))  # Force fallback to example data
