# sentiment_app/models.py

from django.db import models


class AnalysisProject(models.Model):
    """Private analysis workspace owned by one anonymous browser session."""
    name = models.CharField(max_length=120, default='My Feedback Analysis')
    owner_session_key = models.CharField(max_length=40, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-updated_at']

    def __str__(self):
        return self.name


class UploadBatch(models.Model):
    """One CSV upload, retained so reports can be compared later."""
    project = models.ForeignKey(AnalysisProject, on_delete=models.CASCADE, related_name='uploads')
    filename = models.CharField(max_length=255)
    source_column = models.CharField(max_length=120)
    total_rows = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']


class AnalysisRecord(models.Model):
    """Stores the permanent history of every text analyzed."""
    input_text = models.TextField()
    result = models.CharField(max_length=10) # e.g., 'Positive', 'Negative', 'Neutral'
    algorithm = models.CharField(max_length=20) 
    analyzed_at = models.DateTimeField(auto_now_add=True)
    project = models.ForeignKey(
        AnalysisProject, on_delete=models.CASCADE, related_name='records',
        null=True, blank=True,
    )
    upload_batch = models.ForeignKey(
        UploadBatch, on_delete=models.CASCADE, related_name='records',
        null=True, blank=True,
    )

    def __str__(self):
        return f"[{self.analyzed_at.strftime('%Y-%m-%d %H:%M')}] - {self.result} via {self.algorithm}"

    class Meta:
        ordering = ['-analyzed_at']


class WordCount(models.Model):
    """Stores persistent word counts for the word clouds."""
    word = models.CharField(max_length=50) 
    sentiment = models.CharField(max_length=10) # 'Positive' or 'Negative'
    count = models.IntegerField(default=1)
    project = models.ForeignKey(
        AnalysisProject, on_delete=models.CASCADE, related_name='word_counts',
        null=True, blank=True,
    )

    class Meta:
        ordering = ['-count']
        constraints = [
            models.UniqueConstraint(
                fields=['project', 'word', 'sentiment'],
                name='unique_project_sentiment_word',
            ),
        ]

    def __str__(self):
        return f"[{self.sentiment}] {self.word}: {self.count}"
