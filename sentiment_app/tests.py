from unittest.mock import patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import RequestFactory, TestCase

from .models import AnalysisRecord, WordCount
from .views import bulk_analyze_csv, extract_word_cloud_words


class WordCloudTests(TestCase):
    def test_extractor_keeps_short_sentiment_words_and_removes_noise(self):
        self.assertEqual(
            extract_word_cloud_words('I love this good app, but it is bad and sad!'),
            ['love', 'good', 'bad', 'sad'],
        )

    @patch('sentiment_app.views.predict_sentiment')
    def test_manual_analysis_adds_words_to_cloud(self, predict_sentiment):
        predict_sentiment.return_value = ('Positive', ['5.00%', '95.00%'])

        response = self.client.post('/', {'input_text': 'I love this', 'algorithm': 'lr'})

        self.assertEqual(response.status_code, 200)
        self.assertTrue(WordCount.objects.filter(word='love', sentiment='Positive').exists())
        self.assertContains(response, 'love')

    @patch('sentiment_app.views.predict_sentiment')
    def test_csv_analysis_adds_words_to_both_clouds(self, predict_sentiment):
        predict_sentiment.side_effect = [
            ('Positive', ['5.00%', '95.00%']),
            ('Negative', ['95.00%', '5.00%']),
        ]
        upload = SimpleUploadedFile(
            'sentiments.csv',
            b'text\nI love this\nThis is bad and sad\n',
            content_type='text/csv',
        )

        summary = bulk_analyze_csv(upload, 'text', 'lr', RequestFactory().post('/'))

        self.assertEqual(summary['total'], 2)
        self.assertTrue(WordCount.objects.filter(word='love', sentiment='Positive').exists())
        self.assertTrue(WordCount.objects.filter(word='bad', sentiment='Negative').exists())
        self.assertTrue(WordCount.objects.filter(word='sad', sentiment='Negative').exists())
        self.assertEqual(AnalysisRecord.objects.count(), 2)
