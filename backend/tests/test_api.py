import unittest

from fastapi import HTTPException

from backend.app.config import CLINICIAN_PASSWORD, CLINICIAN_USERNAME
from backend.app.main import explain, health, login, logout, metadata, predict, sample_patient
from backend.app.schemas import ExplanationRequest, LoginRequest, PredictionRequest


class AuthTests(unittest.TestCase):
    """Tests for authentication edge cases."""

    def test_login_wrong_password(self) -> None:
        with self.assertRaises(HTTPException) as context:
            login(LoginRequest(username=CLINICIAN_USERNAME, password="wrong-password"))
        self.assertEqual(context.exception.status_code, 401)

    def test_login_wrong_username(self) -> None:
        with self.assertRaises(HTTPException) as context:
            login(LoginRequest(username="notauser", password=CLINICIAN_PASSWORD))
        self.assertEqual(context.exception.status_code, 401)

    def test_logout_invalidates_token(self) -> None:
        session = login(LoginRequest(username=CLINICIAN_USERNAME, password=CLINICIAN_PASSWORD))
        token = f"Bearer {session.access_token}"
        # Token should be valid before logout.
        user = {"username": session.username, "display_name": session.display_name}
        response = metadata(user)
        self.assertIn("selected_genes", response)
        # Log out.
        logout(authorization=token)

    def test_health_no_auth_required(self) -> None:
        # /health must be accessible without a token.
        response = health()
        self.assertEqual(response["status"], "ok")


class ApiTests(unittest.TestCase):
    """Tests for core API functionality."""

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

    def test_metadata_requires_auth(self) -> None:
        # Calling the session manager directly with no token should raise 401.
        with self.assertRaises(HTTPException) as context:
            from backend.app.auth import session_manager
            session_manager.get_user(None)
        self.assertEqual(context.exception.status_code, 401)

    def test_sample_requires_auth(self) -> None:
        from backend.app.auth import session_manager
        with self.assertRaises(HTTPException) as context:
            session_manager.get_user("Bearer invalid-token-xyz")
        self.assertEqual(context.exception.status_code, 401)

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
                PredictionRequest(age=60, gender="male", genes=genes),
                self.user,
            )
        self.assertEqual(context.exception.status_code, 400)

    def test_extra_gene_is_rejected(self) -> None:
        data = metadata(self.user)
        genes = {gene: 0.0 for gene in data["selected_genes"]}
        genes["__FAKE_GENE_XYZ__"] = 1.0  # Inject an unexpected gene.
        with self.assertRaises(HTTPException) as context:
            predict(
                PredictionRequest(age=60, gender="male", genes=genes),
                self.user,
            )
        self.assertEqual(context.exception.status_code, 400)

    def test_predict_age_boundary_min(self) -> None:
        data = metadata(self.user)
        genes = {gene: 0.0 for gene in data["selected_genes"]}
        response = predict(
            PredictionRequest(age=0, gender="male", genes=genes),
            self.user,
        )
        self.assertIn(response.prediction, ["Alive", "Dead"])

    def test_predict_age_boundary_max(self) -> None:
        data = metadata(self.user)
        genes = {gene: 0.0 for gene in data["selected_genes"]}
        response = predict(
            PredictionRequest(age=120, gender="male", genes=genes),
            self.user,
        )
        self.assertIn(response.prediction, ["Alive", "Dead"])


if __name__ == "__main__":
    unittest.main()
