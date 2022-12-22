# LOGS OF TORCHSERVE

(torchserve) ubuntu@ip-172-31-45-124:~/emlov2-session-08$ pytest test_serve/test_
torch_serve-py
==
= test session starts
platform linux - Python 3.9.15, pytest-7.2.0, pluggy-1.0.0
rootdir: /home/ubuntu/emlov2-session-08, configfile: pyproject.toml
plugins: hydra-core-1.2.0
collected 1 item
test_serve/test_torch_serve.py: :TestFargateGradio: :test_predict PASSED
0.19s call
slowest durations
test_serve/test_torch_serve.py::TestFargateGradio::test_predict
(2 durations < 0.005s hidden.
Use
-yv to show these durations-
)
1 passed in 0.28s
[100%]


# LOGS OF GRPC

(torchserve) ubuntu@ip-172-31-45-124:~/emlov2-session-08$ pytest test_serve/test_torch_serve_grpc.py 
================================================== test session starts ==================================================
platform linux -- Python 3.9.15, pytest-7.2.0, pluggy-1.0.0
rootdir: /home/ubuntu/emlov2-session-08, configfile: pyproject.toml
plugins: hydra-core-1.2.0
collected 1 item                                                                                                        

test_serve/test_torch_serve_grpc.py::TestFargateGradio::test_predict FAILED                                       [100%]

======================================================= FAILURES ========================================================
____________________________________________ TestFargateGradio.test_predict _____________________________________________

self = <test_torch_serve_grpc.TestFargateGradio testMethod=test_predict>

    def test_predict(self):
        for image_path in self.image_paths:
            print(f"testing: {image_path}")
    
            response = infer(self.stub, 'cifar', 'test_serve/image/' + image_path)
    
    
            # print(f"response: {response.text}")
    
            data = json.loads(response)
    
            predicted_label = list(data)[0]
            act_label = image_path.split(".")[0].split('_')[-1]
    
            print(f"predicted label: {predicted_label}, actual label: {act_label}")
    
>           self.assertEqual(act_label, predicted_label)
E           AssertionError: 'ship' != 'automobile'
E           - ship
E           + automobile

test_serve/test_torch_serve_grpc.py:55: AssertionError
------------------------------------------------- Captured stdout call --------------------------------------------------
testing: 1000_truck.png
predicted label: truck, actual label: truck
done testing: 1000_truck.png

testing: 10011_cat.png
predicted label: cat, actual label: cat
done testing: 10011_cat.png

testing: 10010_airplane.png
predicted label: airplane, actual label: airplane
done testing: 10010_airplane.png

testing: 10008_airplane.png
predicted label: airplane, actual label: airplane
done testing: 10008_airplane.png

testing: 10001_frog.png
predicted label: frog, actual label: frog
done testing: 10001_frog.png

testing: 10003_ship.png
predicted label: automobile, actual label: ship
=================================================== slowest durations ===================================================
0.12s call     test_serve/test_torch_serve_grpc.py::TestFargateGradio::test_predict

(2 durations < 0.005s hidden.  Use -vv to show these durations.)
================================================ short test summary info ================================================
FAILED test_serve/test_torch_serve_grpc.py::TestFargateGradio::test_predict - AssertionError: 'ship' != 'automobile'
=================================================== 1 failed in 0.36s ===================================================