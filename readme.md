### **Magic Card Search App**

This project is a semantic search application for Magic: The Gathering cards. Instead of relying on exact keywords, it uses **machine learning models** to understand the meaning and function of a card's text, allowing you to find cards with similar effects or concepts.

It processes a comprehensive database of cards to build a powerful search index, letting you find what you're looking for with natural language.

-----

### **Search Capabilities**

The app gives you two ways to search for cards:

  * **By Card Name**: Find other cards that are functionally similar to a specific card (e.g., searching for "Llanowar Elves" will find other mana-generating creatures).
  * **By Description**: Describe a card's effect in your own words (e.g., "creature that gets bigger when attacking") and the app will find the best matches.

-----

### **How It Works**

The application uses two key machine learning models:

1.  **Sentence-BERT (`all-mpnet-base-v2`)**: This model converts the text of every Magic card into a high-dimensional vector (an embedding). Cards with similar meanings are placed close to each other in this vector space.
2.  **Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`)**: After an initial search, this model re-ranks the top results to provide the most relevant matches, ensuring high accuracy.

The embeddings are stored in a **Faiss index**, a highly efficient library for similarity searches, allowing the app to query tens of thousands of cards in milliseconds.

-----

### **Setup and Usage**

For the easiest setup, use Docker. This method ensures all dependencies, including the large card data file, are handled automatically.

#### **Docker Setup**

1.  **Install Docker**: Ensure you have Docker and Docker Compose installed on your machine.
2.  **Build and Run**: From the project's root directory, simply execute the following command:

<!-- end list -->

```bash
docker-compose up
```

The first time you run this, it will take several minutes as it downloads the card database and processes the data. Subsequent runs will be much faster.

Once the process is complete, open your web browser and go to:
[http://localhost:8501](https://www.google.com/search?q=http://localhost:8501)

-----

#### **Manual Setup**

If you prefer not to use Docker, you can set up the project manually.

1.  **Install Python Dependencies**:
    Install all required libraries using `pip`:

    ```bash
    pip install -r requirements.txt
    ```

2.  **Download and Process the Data**:
    Run the `process_data.py` script. This will download the card database and generate the necessary search index files (`unique_cards.pkl` and `cards_faiss.index`).

    ```bash
    python process_data.py
    ```

3.  **Run the Application**:
    Start the Streamlit application with the following command:

    ```bash
    streamlit run app.py
    ```

Once the application is running, open your web browser and navigate to the local address provided in the terminal.