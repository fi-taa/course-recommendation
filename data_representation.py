import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data/coursera-courses.csv', encoding='latin-1',sep=",")


# Plotting a bar chart for 'course_difficulty'
course_difficulty_counts = data['course_difficulty'].value_counts()
course_difficulty_counts.plot(kind='bar', title='Course Difficulty Distribution', xlabel='Difficulty Level', ylabel='Count')
plt.show()

# Plotting a pie chart for 'learning_product_type'
learning_product_type_counts = data['learning_product_type'].value_counts()
learning_product_type_counts.plot(kind='pie', title='Learning Product Type Distribution', autopct='%1.1f%%')
plt.show()


from wordcloud import WordCloud

# Generate a word cloud for 'instructors'
instructors_text = ' '.join(data['instructors'].astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(instructors_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Instructors')
plt.show()
