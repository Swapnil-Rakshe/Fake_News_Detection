import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud


class DataIngestion:
    def __init__(self):
        self.true_data = pd.read_csv("dataset/True.csv")
        self.fake_data = pd.read_csv("dataset/Fake.csv")
    
    def data_visualization(self):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
        
        sns.countplot(y="subject", palette="coolwarm", data=self.true_data, ax=ax1)
        ax1.set_title('True News Subject Distribution')
        
        sns.countplot(y="subject", palette="coolwarm", data=self.fake_data, ax=ax2)
        ax2.set_title('Fake News Subject Distribution')
        
        plt.tight_layout()
        plt.show()
        
    def create_wordcloud(self, data, title):
        all_titles = data.title.str.cat(sep=' ')
        wordcloud = WordCloud(background_color='white', width=800, height=500,
                              max_font_size=180, collocations=False).generate(all_titles)
        plt.figure(figsize=(10,7))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(title, fontsize=20)
        plt.show()
    def visualize_wordclouds(self):
        self.create_wordcloud(self.true_data, 'Real News Title WordCloud')
        self.create_wordcloud(self.fake_data, 'Fake News Title WordCloud')
        
        
if __name__ == "__main__":
    obj = DataIngestion()
    obj.data_visualization()
    obj.visualize_wordclouds()