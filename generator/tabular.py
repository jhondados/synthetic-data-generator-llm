"""LLM + statistical synthetic tabular data generator."""
import pandas as pd
import numpy as np
from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
from langchain_google_vertexai import ChatVertexAI

class SyntheticDataGenerator:
    def __init__(self, method: str = "ctgan"):
        self.method = method
        self.llm = ChatVertexAI(model_name="gemini-1.5-flash-002")

    def fit_and_generate(self, real_df: pd.DataFrame, n_rows: int, pii_columns: list = None) -> pd.DataFrame:
        """Generate synthetic data with same statistical properties."""
        df = real_df.drop(columns=pii_columns or []).copy()
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)
        synthesizer = CTGANSynthesizer(metadata, epochs=300, verbose=True) if self.method == "ctgan" \
            else GaussianCopulaSynthesizer(metadata)
        synthesizer.fit(df)
        synthetic = synthesizer.sample(n_rows)
        return synthetic

    def validate_fidelity(self, real: pd.DataFrame, synthetic: pd.DataFrame) -> dict:
        """Compute statistical fidelity metrics."""
        from sdv.evaluation.single_table import evaluate_quality
        from sdv.metadata import SingleTableMetadata
        meta = SingleTableMetadata(); meta.detect_from_dataframe(real)
        report = evaluate_quality(real, synthetic, meta)
        return {"overall_score": report.get_score(), "column_shapes": report.get_details("Column Shapes").to_dict()}
