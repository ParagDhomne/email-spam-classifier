from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Email Spam Classifier API is running"
    }


def test_spam_email():
    response = client.post(
        "/predict",
        json={
            "text": """
            Congratulations! You have won $1,000,000.
            Click here immediately to claim your prize.
            """
        }
    )

    assert response.status_code == 200
    assert response.json()["prediction"] == "spam"


def test_ham_email():
    response = client.post(
        "/predict",
        json={
            "text": """
            Hi John,
            Can we meet tomorrow at 10 AM to discuss the project?
            """
        }
    )

    assert response.status_code == 200
    assert response.json()["prediction"] == "ham"