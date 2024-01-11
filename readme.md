# Course Recommendation System

This project implements a content-based recommendation system for online courses. The recommendation is based on user input for 'course_provided_by', 'learning_product_type', and 'course_difficulty'. The system uses TF-IDF (Term Frequency-Inverse Document Frequency) and cosine similarity to identify courses that are most similar to the user's preferences.

## Requirements

- Python 3.x
- Pandas
- Scikit-learn
- NLTK (Natural Language Toolkit)

Install the required packages using the following command:

```bash
pip install pandas scikit-learn nltk
```

## usage

### clone the repository

```bash
git clone https://github.com/fi-taa/course-recommendation.git
cd course-recommendation
```

### Run the recommendation system
```bash
python recommendation_system.py
```
### Enter user preferences when prompted:
```bash
Enter 'course_provided_by': Coursera
Enter 'learning_product_type': Course
Enter 'course_difficulty': Intermediate
```

*** View the top 5 recommended courses with course details. ***

##  Customization
Dataset: Replace 'your_data.csv' with your own dataset containing the necessary columns: 'course_provided_by', 'learning_product_type', 'course_difficulty', 'course_name', 'estimated_time_to_complete', 'instructors', 'description'.

User Input: Modify the user_input dictionary in the example usage section according to your preferences.

## Acknowledgments
[coursera-courses.csv](https://github.com/ry05/couReco/blob/master/data/coursera-courses.csv)