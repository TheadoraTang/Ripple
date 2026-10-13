import unittest


class RagEmbeddingDimensionsTest(unittest.TestCase):
    def test_bge_models_do_not_send_dimensions_parameter(self):
        from app.services.rag_service import _embedding_supports_dimensions

        self.assertFalse(_embedding_supports_dimensions("BAAI/bge-m3"))
        self.assertFalse(_embedding_supports_dimensions("bge-m3"))

    def test_text_embedding_models_send_dimensions_parameter(self):
        from app.services.rag_service import _embedding_supports_dimensions

        self.assertTrue(_embedding_supports_dimensions("text-embedding-v4"))

    def test_detects_dimensions_parameter_errors(self):
        from app.services.rag_service import _is_dimensions_parameter_error

        self.assertTrue(_is_dimensions_parameter_error(ValueError("dimensions is not supported")))
        self.assertFalse(_is_dimensions_parameter_error(ValueError("model not found")))


if __name__ == "__main__":
    unittest.main()
