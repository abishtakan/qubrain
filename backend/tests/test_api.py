import unittest

from fastapi import HTTPException

from qubrain.backend.app.config import CLINICIAN_PASSWORD, CLINICIAN_USERNAME
from qubrain.backend.app.main import explain, health, login, metadata, predict, sample_patient
from qubrain.backend.app.schemas import ExplanationRequest, LoginRequest, PredictionRequest


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        session = login(LoginRequest(username=CLINICIAN_USERNAME, password=CLINICIAN_PASSWORD))
        cls.user = {"username": session.username, "display_name": session.display_name}

    def test_health(self) -> None:
        response = health()
        self.assertEqual(response["status"], "ok")

    def test_metadata(self) -> None:
        response = metadata(self.user)
        self.assertGreater(len(response["selected_genes"]), 0)
        self.assertIn("decision_threshold", response)
        self.assertIn("explainability", response)

    def test_predict_with_sample(self) -> None:
        sample = sample_patient(self.user)
        response = predict(
            PredictionRequest(
                age=sample["age"],
                gender=sample["gender"],
                genes=sample["genes"],
            ),
            self.user,
        )
        self.assertGreaterEqual(response.mortality_probability, 0.0)
        self.assertLessEqual(response.mortality_probability, 1.0)
        self.assertGreater(response.decision_threshold, 0.0)
        self.assertLess(response.decision_threshold, 1.0)

    def test_explain_with_sample(self) -> None:
        sample = sample_patient(self.user)
        response = explain(
            ExplanationRequest(
                age=sample["age"],
                gender=sample["gender"],
                genes=sample["genes"],
            ),
            self.user,
        )
        self.assertGreaterEqual(response.mortality_probability, 0.0)
        self.assertLessEqual(response.mortality_probability, 1.0)
        self.assertGreater(len(response.top_risk_increasing) + len(response.top_risk_reducing), 0)

    def test_missing_gene_is_rejected(self) -> None:
        data = metadata(self.user)
        genes = {gene: 0.0 for gene in data["selected_genes"][:-1]}
        with self.assertRaises(HTTPException) as context:
            predict(
                PredictionRequest(
                    age=60,
                    gender="male",
                    genes=genes,
                ),
                self.user,
            )
        self.assertEqual(context.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
