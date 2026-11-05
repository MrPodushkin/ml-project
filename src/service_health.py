"""Container healthcheck over loopback, with no authentication secret required."""

import http.client
import json


def main():
    connection = http.client.HTTPConnection("127.0.0.1", 8000, timeout=3)
    connection.request("GET", "/health")
    response = connection.getresponse()
    result = json.loads(response.read())
    connection.close()
    if response.status != 200 or result.get("status") != "healthy":
        raise SystemExit(1)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
