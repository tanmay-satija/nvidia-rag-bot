import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from chunking import chunk_text, chunk_document
from eval.evaluate import score_retrieval, score_groundedness
from fastapi.testclient import TestClient
from backend.main import app

class CoreTests(unittest.TestCase):
    def test_chunk_overlap_and_invalid_parameters(self):
        self.assertEqual(chunk_text('a b c d e f', 4, 2), ['a b c d', 'c d e f'])
        for size, overlap in [(0, 0), (4, 4), (4, -1)]:
            with self.assertRaises(ValueError):
                chunk_text('text', size, overlap)

    def test_sections_do_not_mix(self):
        chunks = chunk_document('Item 1. Business\nalpha beta\nItem 1A. Risk Factors\ngamma delta', '10-K', '2025-01-26')
        self.assertEqual([c['text'].strip() for c in chunks], ['alpha beta', 'gamma delta'])

    def test_section_apostrophes(self):
        score = score_retrieval({'expected_keywords': ['revenue'], 'expected_section_contains': "Management's Discussion"}, [{'text': 'revenue', 'section': 'Management’s Discussion'}])
        self.assertTrue(score['section_hit'])

    def test_citation_bounds_and_uncited_claims(self):
        score = score_groundedness('Revenue increased by ten percent. Revenue increased by ten percent [3].', 2)
        self.assertEqual(score['uncited_claim_count'], 1)
        self.assertEqual(score['invalid_citations'], [3])

    def test_request_limits(self):
        client = TestClient(app)
        for payload in [{'question': 'test', 'k': 0}, {'question': 'test', 'k': 11}, {'question': ''}]:
            self.assertEqual(client.post('/query', json=payload).status_code, 422)
        self.assertEqual(client.post('/query', json={'question': '   '}).status_code, 400)

    def test_missing_index_error(self):
        with patch('backend.main.get_store', side_effect=FileNotFoundError):
            response = TestClient(app).post('/query', json={'question': 'Revenue?'})
        self.assertEqual(response.status_code, 503)
        self.assertIn('Index not built', response.json()['detail'])

    def test_source_evidence(self):
        from generator import answer_question
        with patch('generator.Groq') as groq, patch.dict('os.environ', {'GROQ_API_KEY': 'test'}):
            groq.return_value.chat.completions.create.return_value.choices[0].message.content = 'Answer [1].'
            result = answer_question('Question', [{'text': 'Evidence', 'form': '10-K', 'report_date': '2025', 'section': 'Business', 'source_url': 'https://www.sec.gov/example'}])
        self.assertEqual(result['sources'][0]['excerpt'], 'Evidence')
        self.assertEqual(result['sources'][0]['url'], 'https://www.sec.gov/example')

if __name__ == '__main__':
    unittest.main()
