# sentiment_app/models.py

from django.db import models

class AnalysisRecord(models.Model):
    """Stores the permanent history of every text analyzed."""
    input_text = models.TextField()
    result = models.CharField(max_length=10) # e.g., 'Positive', 'Negative', 'Neutral'
    algorithm = models.CharField(max_length=20) 
    analyzed_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"[{self.analyzed_at.strftime('%Y-%m-%d %H:%M')}] - {self.result} via {self.algorithm}"

    class Meta:
        ordering = ['-analyzed_at']


class WordCount(models.Model):
    """Stores persistent word counts for the word clouds."""
    word = models.CharField(max_length=50) 
    sentiment = models.CharField(max_length=10) # 'Positive' or 'Negative'
    count = models.IntegerField(default=1)

    class Meta:
        # The combination of word and sentiment must be unique
        unique_together = ('word', 'sentiment') 
        ordering = ['-count']

    def __str__(self):
        return f"[{self.sentiment}] {self.word}: {self.count}"