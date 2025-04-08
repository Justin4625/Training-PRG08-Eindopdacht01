import kNear from "./knear.js"
import { HandLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";

const k = 1
const machine = new kNear(k);

const enableWebcamButton = document.getElementById("webcamButton")
const exportButton = document.getElementById("exportButton")

const video = document.getElementById("webcam")
const canvasElement = document.getElementById("output_canvas")
const canvasCtx = canvasElement.getContext("2d")

const drawUtils = new DrawingUtils(canvasCtx)
let handLandmarker = undefined;
let webcamRunning = false;
let results = undefined;

let image = document.querySelector("#myimage")

let keysPressed = {};

document.addEventListener("keydown", (event) => {
    keysPressed[event.key] = true;
});

document.addEventListener("keyup", (event) => {
    keysPressed[event.key] = false;
});

function isKeyPressed(key) {
    return keysPressed[key] === true;
}

const createHandLandmarker = async () => {
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 1
    });

    enableWebcamButton.addEventListener("click", (e) => enableCam(e))
    exportButton.addEventListener("click", (e) => exportTrainingData(e))
}

async function enableCam() {
    webcamRunning = true;
    try {
        const stream = await navigator.mediaDevices.getUserMedia({video: true, audio: false});
        video.srcObject = stream;
        video.addEventListener("loadeddata", () => {
            canvasElement.style.width = video.videoWidth;
            canvasElement.style.height = video.videoHeight;
            canvasElement.width = video.videoWidth;
            canvasElement.height = video.videoHeight;
            document.querySelector(".videoView").style.height = video.videoHeight + "px";
            predictWebcam();
        });
    } catch (error) {
        console.error("Error accessing webcam:", error);
    }
}

function storeHandData(label, handData) {
    let storedData = JSON.parse(localStorage.getItem('handData')) || {};
    if (!storedData[label]) {
        storedData[label] = [];
    }
    storedData[label].push(handData);
    localStorage.setItem('handData', JSON.stringify(storedData));
}

function loadHandData() {
    let storedData = JSON.parse(localStorage.getItem('handData')) || {};
    for (let label in storedData) {
        storedData[label].forEach(data => {
            machine.learn(data, label);
        });
    }
}

function exportTrainingData() {
    let storedData = localStorage.getItem('handData');
    if (storedData) {
        let blob = new Blob([storedData], { type: 'application/json' });
        let url = URL.createObjectURL(blob);
        let a = document.createElement('a');
        a.href = url;
        a.download = 'TrainingData.json';
        a.click();
        URL.revokeObjectURL(url);
    } else {
        console.log("No training data to export.");
    }
}

window.addEventListener('load', loadHandData);

async function predictWebcam() {
    results = await handLandmarker.detectForVideo(video, performance.now());

    let predictionDiv = document.getElementById('predictionDiv');

    if (results.landmarks.length === 0) {
        predictionDiv.innerHTML = '';
    } else {
        let hand = results.landmarks[0];
        if (hand) {
            let thumb = hand[4];
            image.style.transform = `translate(${video.videoWidth - thumb.x * video.videoWidth}px, ${thumb.y * video.videoHeight}px)`;
        }

        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        for (let hand of results.landmarks) {
            console.log(hand[4])
            drawUtils.drawConnectors(hand, HandLandmarker.HAND_CONNECTIONS, {color: "#00FF00", lineWidth: 5});
            drawUtils.drawLandmarks(hand, {radius: 4, color: "#FF0000", lineWidth: 2});

            let handData = hand.map((o) => [o.x, o.y, o.z]).flat();

            if (isKeyPressed("1")) {
                machine.learn(handData, "Zwaard");
                storeHandData("Zwaard", handData);
            }

            if (isKeyPressed("2")) {
                machine.learn(handData, "Schild");
                storeHandData("Schild", handData);
            }

            if (isKeyPressed("3")) {
                machine.learn(handData, "Charge");
                storeHandData("Charge", handData);
            }

            let prediction = machine.classify(handData);

            if (prediction) {
                switch (prediction) {
                    case "Zwaard":
                        predictionDiv.innerHTML = 'Zwaard 🗡️';
                        break;
                    case "Schild":
                        predictionDiv.innerHTML = 'Schild 🛡️';
                        break;
                    case "Charge":
                        predictionDiv.innerHTML = 'Charge ⚡';
                        break;
                    default:
                        predictionDiv.innerHTML = '';
                        break;
                }
            } else {
                predictionDiv.innerHTML = '';
            }
        }
    }

    if (webcamRunning) {
        window.requestAnimationFrame(predictWebcam);
    }
}

if (navigator.mediaDevices?.getUserMedia) {
    createHandLandmarker()
}
