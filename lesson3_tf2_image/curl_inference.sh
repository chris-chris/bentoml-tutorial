echo '{"instances":[{"b64":"'$(base64 test.png)'"}]}' > test.json
curl -vX POST http://localhost:5000/predict -d @test.json --header "Content-Type: application/json"