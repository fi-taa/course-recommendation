import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.tokenize import sent_tokenize 
import nltk
nltk.download('punkt')

data = pd.read_csv('data/coursera-courses.csv', encoding='latin-1',sep=",")

relevant_columns = ['course_provided_by', 'learning_product_type', 'course_difficulty', 'course_name', 'estimated_time_to_complete', 'instructors', 'description']
data = data[relevant_columns]

# Combine relevant columns into a single feature column for TF-IDF
data['combined_features'] = data['course_provided_by'] + ' ' + data['learning_product_type'] + ' ' + data['course_difficulty']

# Create a TfidfVectorizer to convert the combined features into TF-IDF features
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data['combined_features'].fillna(''))

# Calculate cosine similarity
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get content-based recommendations
def user_based_content_recommendations(user_input):
    # Create a new DataFrame with the user's input
    user_data = pd.DataFrame([user_input], columns=['course_provided_by', 'learning_product_type', 'course_difficulty'])

    # Combine user input into a single feature column
    user_data['combined_features'] = user_data['course_provided_by'] + ' ' + user_data['learning_product_type'] + ' ' + user_data['course_difficulty']

    # Convert user input into TF-IDF features
    user_tfidf_matrix = tfidf_vectorizer.transform(user_data['combined_features'].fillna(''))

    # Calculate cosine similarity between user input and all courses
    user_cosine_similarities = linear_kernel(user_tfidf_matrix, tfidf_matrix)

    # Get the indices of courses sorted by similarity
    sim_indices = user_cosine_similarities.argsort()[0][::-1][1:6]

    # Return the top 5 recommended courses
    recommendations = data.iloc[sim_indices][['course_name', 'estimated_time_to_complete', 'instructors', 'description']]

    # Extract only the first 5 sentences from each description
    recommendations['description'] = recommendations['description'].apply(lambda x: ' '.join(sent_tokenize(x)[:5]))

    return recommendations

# Function to print formatted recommendations
def print_formatted_recommendations(recommendations):
    print("\nTop 5 Recommended Courses:")
    print("=" * 50)
    for i, (index, row) in enumerate(recommendations.iterrows(), start=1):
        print(f"{i}. Course Name: {row['course_name']}")
        print(f"   Estimated Time to Complete: {row['estimated_time_to_complete']}")
        print(f"   Instructors: {row['instructors']}")
        print(f"   Description: {row['description']}\n")
    print("=" * 50)

# Example usage with user input
user_input = {'course_provided_by': 'Coursera', 'learning_product_type': 'Course', 'course_difficulty': 'Intermediate'}
recommendations = user_based_content_recommendations(user_input)

# Print the formatted recommendations
print_formatted_recommendations(recommendations)
