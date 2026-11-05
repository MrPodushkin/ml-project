"""Real HTTP smoke checks executed inside the built non-root service container."""

import http.client
import json
import os


def main():

    def request(method, path, body=None, auth=False):
        connection = http.client.HTTPConnection("127.0.0.1", 8000, timeout=5)
        headers = {"Content-Type": "application/json"}
        if auth:
            headers["Authorization"] = "Bearer " + os.environ["LAB4_API_TOKEN"]
        connection.request(
            method, path, body=json.dumps(body) if body is not None else None, headers=headers
        )
        response = connection.getresponse()
        raw = response.read()
        connection.close()
        return (response.status, raw)

    if not os.getuid() == 10001:
        raise RuntimeError("Container HTTP smoke check failed")
    status, raw = request("GET", "/health")
    if not (status == 200 and json.loads(raw)["status"] == "healthy"):
        raise RuntimeError("Container HTTP smoke check failed")
    if not request("GET", "/")[0] == 200:
        raise RuntimeError("Container HTTP smoke check failed")
    if not request("POST", "/predict", {"message": "hello"})[0] == 401:
        raise RuntimeError("Container HTTP smoke check failed")
    status, raw = request("POST", "/predict", {"message": "FREE cash prize! Claim now!!!"}, True)
    if not (status == 200 and json.loads(raw)["prediction"] == "spam"):
        raise RuntimeError("Container HTTP smoke check failed")
    if not request("POST", "/predict", {"message": ""}, True)[0] == 422:
        raise RuntimeError("Container HTTP smoke check failed")
    if not request("GET", "/metrics")[0] == 401:
        raise RuntimeError("Container HTTP smoke check failed")
    if not request("GET", "/metrics", auth=True)[0] == 200:
        raise RuntimeError("Container HTTP smoke check failed")
    print(
        json.dumps(
            {
                "uid": os.getuid(),
                "health": "healthy",
                "prediction": "spam",
                "unauthorized_status": 401,
                "invalid_status": 422,
                "smoke_passed": True,
            }
        )
    )


if __name__ == "__main__":
    main()
