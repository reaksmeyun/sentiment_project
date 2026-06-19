from django.db import migrations


def clear_legacy_word_counts(apps, schema_editor):
    """Remove counts created by the old whole-review classification rule."""
    WordCount = apps.get_model('sentiment_app', 'WordCount')
    WordCount.objects.all().delete()


class Migration(migrations.Migration):
    dependencies = [
        ('sentiment_app', '0002_alter_analysisrecord_options_alter_wordcount_options_and_more'),
    ]

    operations = [
        migrations.RunPython(clear_legacy_word_counts, migrations.RunPython.noop),
    ]
