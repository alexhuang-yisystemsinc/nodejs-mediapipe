/*
 * Copyright 2024 Forrest Moulin
 *
 * Portions of this code are based on MediaPipe code:
 * Copyright 2023 The MediaPipe Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * hand_face_detection.js
 */

  // Import required vision module from MediaPipe using CDN
  import vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";
  // Extract required classes from vision module
  const { FaceLandmarker, FilesetResolver, DrawingUtils, GestureRecognizer } = vision;

  let gestureNameMap = {};
  let faceLandmarker;
  let gestureRecognizer;
  let webcamRunning = false;
  let handGestureRunning = false;
  let delegateType = 'GPU';
  const video = document.getElementById("webcam");
  const canvasElement = document.getElementById("output_canvas");
  const canvasCtx = canvasElement.getContext("2d");
  const enableWebcamButton = document.getElementById("webcamButton");
  const gestureButton = document.getElementById("gestureButton");
  const gestureOutput = document.getElementById("gesture_output");
  const confidenceOutput = document.getElementById("confidence_output");
  const handednessOutput = document.getElementById("handedness_output");
  const faceOutput = document.getElementById("face_output");
  const handCountOutput = document.getElementById("hand_count_output");
  const LEFT_EYE_LANDMARKS = [463, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374,
  380, 381, 382, 362];// Left eye landmarks

  const RIGHT_EYE_LANDMARKS = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145,
  144, 163, 7];  // Right eye landmarks

const LEFT_IRIS_LANDMARKS = [474, 475, 477, 476];  // Left iris landmarks
const RIGHT_IRIS_LANDMARKS = [469, 470, 471, 472];  // Right iris landmarks

  const NOSE_LANDMARKS = [193, 168, 417, 122, 351, 196, 419, 3, 248, 236, 456, 198, 420, 131, 360, 
                          49, 279, 48, 278, 219, 439, 59, 289, 218, 438, 237, 457, 44, 19, 274];  // Nose landmarks
  const NOSE_LANDMARKS2 = [168, 417, 351, 419, 248, 456, 420, 360, 
                          279, 278, 439, 289, 438, 457, 274, 19, 44, 237, 218, 
                          59, 219, 48, 49, 131, 198, 236, 3, 196, 122, 193]; 
  const MOUTH_LANDMARKS = [0, 267, 269, 270, 409, 306, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39,
  37];  // Mouth landmarks
  var noseConnection = [];
  var mouseConnection= [];
  function makeConnection(points, bClose, symSource) {
    var connections = [];
    if (symSource) {
      var half = points.length / 2;
      for (var i = 0; i < (half - 1); i++) {
        var pair = { start: points[i * 2], end: points[(i + 1) * 2] };
        connections.push(pair);
      }
      if (bClose) {
        var pair = { start: points[(half - 1) * 2], end: points[(half - 1) * 2 + 1] };
        connections.push(pair);
      }
      for (var i = half - 1; i >= 1; i--) {
        var pair = { start: points[i * 2 + 1], end: points[(i - 1) * 2 + 1] };
        connections.push(pair);
      }
      if (bClose) {
        var pair = { start: points[1], end: points[0] };
        connections.push(pair);
      }
    } else {
      for (var i = 0; i < points.length - 1; i++) {
        var pair = { start: points[i], end: points[i + 1] };
        connections.push(pair);
      }
      if (bClose) {
        pair = { start: points[points.length - 1], end: points[0] };
        connections.push(pair);
      }
    }

    return connections;
  }
  mouseConnection = makeConnection(MOUTH_LANDMARKS, true, true);
  noseConnection = makeConnection(NOSE_LANDMARKS2, true, false);

  function labelLandmarks(canvasContext, landmarks, width, height) {

    canvasContext.font = "10px serif";
    var labelIndex = 0;
    for (var pt of landmarks) {
      console.log(pt);
      canvasContext.strokeText(labelIndex.toString(), pt.x * width, pt.y * height);
      labelIndex++;
    }
  }
  function makeLandmarkPath(landmarks, lineSegments, width, height) {
    var newPath = new Path2D();
    var startPtIndex = lineSegments[0].start;
    var endPtIndex = lineSegments[0].start;

    newPath.moveTo(landmarks[startPtIndex].x * width, landmarks[startPtIndex].y * height);
    for (var step of lineSegments) {
      newPath.lineTo(landmarks[step.end].x * width, landmarks[step.end].y * height);
    }
    newPath.lineTo(landmarks[endPtIndex].x * width, landmarks[endPtIndex].y * height);
    return newPath;
  }
  function makeMousePath(landmarks, lineSegments, width, height) {
    var newPath = new Path2D();
    // 0 - 9: outside/lower lip line
    // 10 - 19: outside/upper lip line
    // 20 - 29: insider/lower
    // 30 - 39: inside/upper
    var startPtIndex = lineSegments[0].start;
    var endPtIndex = lineSegments[0].start;
    newPath.moveTo(landmarks[startPtIndex].x * width, landmarks[startPtIndex].y * height)
    var ptIdx = 0;
    for (var i = 0; i < 10; i++) {
      ptIdx = lineSegments[i].end;
      newPath.lineTo(landmarks[ptIdx].x * width, landmarks[ptIdx].y * height)
    }
    for (var i = 19; i >=10; i--) {
      ptIdx = lineSegments[i].start;
      newPath.lineTo(landmarks[ptIdx].x * width, landmarks[ptIdx].y * height)
    }
    startPtIndex = lineSegments[20].start;
    endPtIndex = lineSegments[20].start;
    newPath.moveTo(landmarks[startPtIndex].x * width, landmarks[startPtIndex].y * height)
    for (var i = 20; i < 30; i++) {
      ptIdx = lineSegments[i].end;
      newPath.lineTo(landmarks[ptIdx].x * width, landmarks[ptIdx].y * height)
    }
    for (var i = 39; i >= 30; i--) {
      ptIdx = lineSegments[i].start;
      newPath.lineTo(landmarks[ptIdx].x * width, landmarks[ptIdx].y * height)
    }
    return newPath;
  }

  async function createFaceLandmarker() {
    const filesetResolver = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
    );
    faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
        delegate: delegateType // "GPU" or "CPU"
      },
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true,
      runningMode: "VIDEO",
      numFaces: 1
    });
  }

  async function createGestureRecognizer() {
    const filesetResolver = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
    );
    gestureRecognizer = await GestureRecognizer.createFromOptions(filesetResolver, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
        delegate: delegateType //"GPU" pr CPU
      },
      runningMode: "VIDEO", 
      numHands: 2
    });
  }

  async function loadGestureNameMap() {
    try {
      const response = await fetch('/public/json/gesture_map.json');
      
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      gestureNameMap = await response.json();
      console.log("Gesture name map loaded successfully:", gestureNameMap);
      
    } catch (error) {
      console.error("Error loading gesture name map:", error);
    }
  }

  function sendGestureToServer(gesture) {
    fetch('/save-gesture', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ gesture: gesture })
    })
    .then(response => response.json())
    .then(data => {
        if (data.errors) {
            console.error('Validation errors:', data.errors);
        } else {
            console.log(data.message);
        }
    })
    .catch(error => console.error('Error:', error));
  }

  loadGestureNameMap();
  createFaceLandmarker();
  createGestureRecognizer();

  enableWebcamButton.addEventListener("click", enableCam);
  gestureButton.addEventListener("click", toggleHandGestureDetection);

  function enableCam() {
    if (!faceLandmarker || !gestureRecognizer) {
      console.log("Wait! Models not loaded yet.");
      return;
    }

    webcamRunning = !webcamRunning;
    enableWebcamButton.innerText = webcamRunning ? "DISABLE FACE" : "DETECT FACE";
    gestureButton.disabled = !webcamRunning;

    const constraints = {
      video: { width: 1280, height: 720 }
    };

    if (webcamRunning) {
      navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
        video.srcObject = stream;
        video.addEventListener("loadeddata", predictWebcam);
      });
    } else {
      const stream = video.srcObject;
      if (stream) {
        const tracks = stream.getTracks();
        tracks.forEach(track => track.stop());
        video.srcObject = null;
      }
      canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    }
  }

  function toggleHandGestureDetection() {
    handGestureRunning = !handGestureRunning;
    gestureButton.innerText = handGestureRunning ? "DISABLE HANDS" : "DETECT HANDS";
  }

  function updateCanvasSize() {
    const videoRatio = video.videoHeight / video.videoWidth;
    video.style.width = '100%';
    video.style.height = 'auto';
    canvasElement.style.width = '100%';
    canvasElement.style.height = 'auto';
    canvasElement.width = video.videoWidth;
    canvasElement.height = video.videoHeight;
  }
  function drawCurve(ctx, ptsa, tension, isClosed, numOfSegments, showPoints) {

    //ctx.beginPath();

    drawLines(ctx, getCurvePoints(ptsa, tension, isClosed, numOfSegments));
  /*
    if (showPoints) {
      canvasTx.beginPath();
      for (var i = 0; i < ptsa.length - 1; i += 2)
        canvasTx.rect(ptsa[i] - 2, ptsa[i + 1] - 2, 4, 4);
    }

    ctx.stroke();
    */
  }


  var tension = 1;

  //drawCurve(ctx, myPoints); //default tension=0.5
  //drawCurve(ctx, myPoints, tension);


  function getCurvePoints(pts, tension, isClosed, numOfSegments) {

    // use input value if provided, or use a default value     
    tension = (typeof tension != 'undefined') ? tension : 0.5;
    isClosed = isClosed ? isClosed : false;
    numOfSegments = numOfSegments ? numOfSegments : 16;

    var _pts = [], res = [],  // clone array
      x, y,         // our x,y coords
      t1x, t2x, t1y, t2y,   // tension vectors
      c1, c2, c3, c4,       // cardinal points
      st, t, i;     // steps based on num. of segments

    // clone array so we don't change the original
    //
    _pts = pts.slice(0);

    // The algorithm require a previous and next point to the actual point array.
    // Check if we will draw closed or open curve.
    // If closed, copy end points to beginning and first points to end
    // If open, duplicate first points to befinning, end points to end
    if (isClosed) {
      _pts.unshift(pts[pts.length - 1]);
      _pts.unshift(pts[pts.length - 2]);
      _pts.unshift(pts[pts.length - 1]);
      _pts.unshift(pts[pts.length - 2]);
      _pts.push(pts[0]);
      _pts.push(pts[1]);
    }
    else {
      _pts.unshift(pts[1]);   //copy 1. point and insert at beginning
      _pts.unshift(pts[0]);
      _pts.push(pts[pts.length - 2]); //copy last point and append
      _pts.push(pts[pts.length - 1]);
    }

    // ok, lets start..

    // 1. loop goes through point array
    // 2. loop goes through each segment between the 2 pts + 1e point before and after
    for (i = 2; i < (_pts.length - 4); i += 2) {
      for (t = 0; t <= numOfSegments; t++) {

        // calc tension vectors
        t1x = (_pts[i + 2] - _pts[i - 2]) * tension;
        t2x = (_pts[i + 4] - _pts[i]) * tension;

        t1y = (_pts[i + 3] - _pts[i - 1]) * tension;
        t2y = (_pts[i + 5] - _pts[i + 1]) * tension;

        // calc step
        st = t / numOfSegments;

        // calc cardinals
        c1 = 2 * Math.pow(st, 3) - 3 * Math.pow(st, 2) + 1;
        c2 = -(2 * Math.pow(st, 3)) + 3 * Math.pow(st, 2);
        c3 = Math.pow(st, 3) - 2 * Math.pow(st, 2) + st;
        c4 = Math.pow(st, 3) - Math.pow(st, 2);

        // calc x and y cords with common control vectors
        x = c1 * _pts[i] + c2 * _pts[i + 2] + c3 * t1x + c4 * t2x;
        y = c1 * _pts[i + 1] + c2 * _pts[i + 3] + c3 * t1y + c4 * t2y;

        //store points in array
        res.push(x);
        res.push(y);

      }
    }

    return res;
  }

  function drawLines(path2D, pts) {
    path2D.moveTo(pts[0], pts[1]);
    for (var i = 2; i < pts.length - 1; i += 2) {
      path2D.lineTo(pts[i], pts[i + 1]);
    }
  }

  function makeMousePathBCurve(canvasCtx, existingPath, landmarks, lineSegments, width, height) {
    var newPath = null;

    if (existingPath != null) {
      newPath = existingPath;
    } else {
      newPath = new Path2D();
    }
    var myPoints = [];
    var myPoints2 = [];
    // 0 - 9: outside/lower lip line
    // 10 - 19: outside/upper lip line
    // 20 - 29: insider/lower
    // 30 - 39: inside/upper
    var startPtIndex = lineSegments[0].start;
    var endPtIndex = lineSegments[0].start;
    var ptIdx = 0;
    newPath.moveTo(landmarks[startPtIndex].x * width, landmarks[startPtIndex].y * height);
    myPoints = [landmarks[startPtIndex].x * width, landmarks[startPtIndex].y * height];

    for (var i = 0; i < 10; i++) {
      var pt = lineSegments[i].end;

      myPoints.push(landmarks[pt].x * width);
      myPoints.push(landmarks[pt].y * height);
    }

    for (var i = 19; i >= 10; i--) {
      var ptIdx = lineSegments[i].start;
      //newPath.lineTo(landmarks[ptIdx].x * width, landmarks[ptIdx].y * height);
      myPoints.push(landmarks[ptIdx].x * width);
      myPoints.push(landmarks[ptIdx].y * height);
    }
    drawCurve(newPath, myPoints, 0.5);


    startPtIndex = lineSegments[20].start;
    endPtIndex = lineSegments[20].start;
    newPath.moveTo(landmarks[startPtIndex].x * width, landmarks[startPtIndex].y * height);
    myPoints2 = [landmarks[startPtIndex].x * width, landmarks[startPtIndex].y * height];

    for (var i = 20; i < 30; i++) {
      var pt = lineSegments[i].end;

      myPoints2.push(landmarks[pt].x * width);
      myPoints2.push(landmarks[pt].y * height);
    }

    for (var i = 39; i >= 30; i--) {
      var ptIdx = lineSegments[i].start;
      //newPath.lineTo(landmarks[ptIdx].x * width, landmarks[ptIdx].y * height);
      myPoints2.push(landmarks[ptIdx].x * width);
      myPoints2.push(landmarks[ptIdx].y * height);
    }
    drawCurve(newPath, myPoints2, 0.5);

    return newPath;
  }

  function makeMouthInsideCurve(canvasCtx, existingPath, landmarks, lineSegments, width, height) {
    var newPath = null;

    if (existingPath != null) {
      newPath = existingPath;
    } else {
      newPath = new Path2D();
    }
    var myPoints = [];
    var myPoints2 = [];
    // 0 - 9: outside/lower lip line
    // 10 - 19: outside/upper lip line
    // 20 - 29: insider/lower
    // 30 - 39: inside/upper
    var startPtIndex = lineSegments[0].start;
    var endPtIndex = lineSegments[0].start;
    var ptIdx = 0;


    startPtIndex = lineSegments[20].start;
    endPtIndex = lineSegments[20].start;
    newPath.moveTo(landmarks[startPtIndex].x * width, landmarks[startPtIndex].y * height);
    myPoints2 = [landmarks[startPtIndex].x * width, landmarks[startPtIndex].y * height];

    for (var i = 20; i < 30; i++) {
      var pt = lineSegments[i].end;

      myPoints2.push(landmarks[pt].x * width);
      myPoints2.push(landmarks[pt].y * height);
    }

    for (var i = 39; i >= 30; i--) {
      var ptIdx = lineSegments[i].start;
      //newPath.lineTo(landmarks[ptIdx].x * width, landmarks[ptIdx].y * height);
      myPoints2.push(landmarks[ptIdx].x * width);
      myPoints2.push(landmarks[ptIdx].y * height);
    }
    drawCurve(newPath, myPoints2, 0.5);

    return newPath;
  }

  function makeFaceOvalCurve(canvasCtx, existingPath, landmarks, lineSegments, width, height) {
    var newPath = null;
    
    if (existingPath != null) {
      newPath = existingPath;
    } else{
      newPath = new Path2D();
    }
      var myPoints = [];
    var myPoints2 = [];
    // 0 - 9: outside/lower lip line
    // 10 - 19: outside/upper lip line
    // 20 - 29: insider/lower
    // 30 - 39: inside/upper
    var startPtIndex = lineSegments[0].start;
    var endPtIndex = lineSegments[0].start;
    var ptIdx = 0;
    newPath.moveTo(landmarks[startPtIndex].x * width, landmarks[startPtIndex].y * height);
    myPoints = [landmarks[startPtIndex].x * width, landmarks[startPtIndex].y * height];

    for (var i = 0; i < lineSegments.length; i++) {
      var pt = lineSegments[i].end;

      myPoints.push(landmarks[pt].x * width);
      myPoints.push(landmarks[pt].y * height);
    }

    drawCurve(newPath, myPoints, 0.5);

    return newPath;
  }

  function makeEyeCurve(canvasCtx, existingPath, landmarks, lineSegments, width, height) {
    var newPath = null;
    
    if (existingPath != null) {
      newPath = existingPath;
    } else{
      newPath = new Path2D();
    }
      var myPoints = [];
    var myPoints2 = [];
    // 0 - 9: outside/lower lip line
    // 10 - 19: outside/upper lip line
    // 20 - 29: insider/lower
    // 30 - 39: inside/upper
    var startPtIndex = lineSegments[0].start;
    var endPtIndex = lineSegments[0].start;
    var ptIdx = 0;
    newPath.moveTo(landmarks[startPtIndex].x * width, landmarks[startPtIndex].y * height);
    myPoints = [landmarks[startPtIndex].x * width, landmarks[startPtIndex].y * height];

    for (var i = 0; i < lineSegments.length; i++) {
      var pt = lineSegments[i].end;

      myPoints.push(landmarks[pt].x * width);
      myPoints.push(landmarks[pt].y * height);
    }

    drawCurve(newPath, myPoints, 0.5);

    return newPath;
  }


  function makeEyeBrowCurve(canvasCtx, existingPath, landmarks, lineSegments, width, height) {
    var newPath = null;

    if (existingPath != null) {
      newPath = existingPath;
    } else {
      newPath = new Path2D();
    }
    var myPoints = [];
    const halfSize = lineSegments.length / 2;
    // 0 - 9: outside/lower lip line
    // 10 - 19: outside/upper lip line
    // 20 - 29: insider/lower
    // 30 - 39: inside/upper
    var startPtIndex = lineSegments[0].start;
    var endPtIndex = lineSegments[0].start;
    var ptIdx = 0;
    newPath.moveTo(landmarks[startPtIndex].x * width, landmarks[startPtIndex].y * height);
    myPoints = [landmarks[startPtIndex].x * width, landmarks[startPtIndex].y * height];

    for (var i = 0; i < halfSize; i++) {
      var pt = lineSegments[i].end;

      myPoints.push(landmarks[pt].x * width);
      myPoints.push(landmarks[pt].y * height);
    }

    for (var i = lineSegments.length - 1; i >= halfSize; i--) {
      var ptIdx = lineSegments[i].start;
      //newPath.lineTo(landmarks[ptIdx].x * width, landmarks[ptIdx].y * height);
      myPoints.push(landmarks[ptIdx].x * width);
      myPoints.push(landmarks[ptIdx].y * height);
    }
    ptIdx = lineSegments[halfSize].end;
    //newPath.lineTo(landmarks[ptIdx].x * width, landmarks[ptIdx].y * height);
    myPoints.push(landmarks[ptIdx].x * width);
    myPoints.push(landmarks[ptIdx].y * height);

    ptIdx = lineSegments[0].start;
    //newPath.lineTo(landmarks[ptIdx].x * width, landmarks[ptIdx].y * height);
    myPoints.push(landmarks[ptIdx].x * width);
    myPoints.push(landmarks[ptIdx].y * height);

    drawCurve(newPath, myPoints, 0.5);

    return newPath;
  }
  async function predictWebcam() {
    updateCanvasSize();

    if (webcamRunning) {
      const startTimeMs = performance.now();
      const faceResults = await faceLandmarker.detectForVideo(video, startTimeMs);
      console.log(faceResults);
      canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

      if (faceResults.faceLandmarks) {
        const drawingUtils = new DrawingUtils(canvasCtx);
        faceOutput.innerText = "Face landmarks detected.";
        for (const landmarks of faceResults.faceLandmarks) {
          console.log(landmarks);

          /*
          drawingUtils.drawLandmarks(landmarks,
            { color: "#C0C0C070", lineWidth: 0.5 }
          );
          */
         /*
          drawingUtils.drawConnectors(
            landmarks,
            FaceLandmarker.FACE_LANDMARKS_TESSELATION,
            { color: "#C0C0C070", lineWidth: 1 }
          );
          */
         /*
          drawingUtils.drawConnectors(
            landmarks,
            FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW,
            { color: "#83f47e" } // Right eyebrow color (#FF3030 is default) ff5722 is orange
          );
          */
          /*
          drawingUtils.drawConnectors(
            landmarks,
            FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
            { color: "#83f47e" } // Right eye color (#FF3030 is default) ff5722 is orange
          );
          drawingUtils.drawConnectors(
            landmarks,
            FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS,
            { color: "#83f47e" } // Right iris color (#FF3030 is default) ff5722 is orange
          );
          */

          console.log("FACE_LANDMARKS_LEFT_EYEBROW");
          console.log(FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW);
          /*
          drawingUtils.drawConnectors(
            landmarks,
            FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW,
            { color: "#83f47e" } // Green left eyebrow color (#30FF30 is default)
          );
          */
          /*
          console.log("FACE_LANDMARKS_LEFT_EYE");
          console.log(FaceLandmarker.FACE_LANDMARKS_LEFT_EYE);
          drawingUtils.drawConnectors(
            landmarks,
            FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,
            { color: "#83f47e" } // Green left eye color (#30FF30 is default)
          );
          console.log("FACE_LANDMARKS_LEFT_IRIS");
          console.log(FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS);
          drawingUtils.drawConnectors(
            landmarks,
            FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS,
            { color: "#83f47e" } // Green left iris color (#30FF30 is default)
          );
          */
          /*
          drawingUtils.drawConnectors(
            landmarks,
            FaceLandmarker.FACE_LANDMARKS_CONTOURS,
            { color: "#A0A0A0" } // face outline color
          );
          */
          console.log("FACE_LANDMARKS_LIPS");
          console.log(FaceLandmarker.FACE_LANDMARKS_LIPS);

          /*
          drawingUtils.drawConnectors(
            landmarks,
            noseConnection,
            { color: "#FF0000" } // Lips color (#E0E0E0 is default)
          );
          */
          //labelLandmarks(canvasCtx, landmarks, video.videoWidth, video.videoHeight);


          var mouthPath2D = makeMousePathBCurve(canvasCtx, null, landmarks, FaceLandmarker.FACE_LANDMARKS_LIPS, video.videoWidth, video.videoHeight);
          
          var mouthPath2DInside = makeMouthInsideCurve(canvasCtx, null, landmarks, FaceLandmarker.FACE_LANDMARKS_LIPS, video.videoWidth, video.videoHeight);

          var leftEyePath2D = makeEyeBrowCurve(canvasCtx, null, landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, video.videoWidth, video.videoHeight);
          var rightEyePath2D = makeEyeBrowCurve(canvasCtx, null, landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, video.videoWidth, video.videoHeight);

          var rightEyeBrowPath2D = makeEyeBrowCurve(canvasCtx, null, landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW, video.videoWidth, video.videoHeight);
          var leftEyeBrowPath2D = makeEyeBrowCurve(canvasCtx, null, landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW, video.videoWidth, video.videoHeight);

          //canvasCtx.fillRect(20, 20, 160, 160);
          var faceOval2D = makeFaceOvalCurve(canvasCtx, null, landmarks, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, video.videoWidth, video.videoHeight);
          faceOval2D.addPath(mouthPath2D);
          faceOval2D.addPath(mouthPath2DInside);
          faceOval2D.addPath(leftEyePath2D);
          faceOval2D.addPath(rightEyePath2D);
          faceOval2D.addPath(rightEyeBrowPath2D);
          faceOval2D.addPath(leftEyeBrowPath2D);
          canvasCtx.filter = "blur(2px)";

          canvasCtx.fillStyle = "rgb(255 200 200 / 30%)";//gradient;//"blue";
          canvasCtx.fill(faceOval2D, "evenodd");
          //canvasCtx.color = "rgb(255, 255, 255)";
          //canvasCtx.stroke(mouthPath2D);
          canvasCtx.fillStyle = "rgb(144 72 72 / 40%)";//gradient;//"blue";
          canvasCtx.fill(mouthPath2D, "evenodd");
          console.log("NOSE-Landmark");
          console.log(noseConnection);
          var nosePath2D = makeEyeCurve(canvasCtx, null, landmarks, noseConnection, video.videoWidth, video.videoHeight);
          canvasCtx.stroke(nosePath2D);
          canvasCtx.fillRect(0, 0, 200, 200);

          console.log("FACE_LANDMARKS_FACE_OVAL");
          console.log(FaceLandmarker.FACE_LANDMARKS_FACE_OVAL);
          console.log("FACE_LANDMARKS_CONTOURS");
          console.log(FaceLandmarker.FACE_LANDMARKS_CONTOURS);
          console.log("FACE_LANDMARKS_RIGHT_EYE");
          console.log(FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE);
          console.log("FACE_LANDMARKS_LEFT_EYE");
          console.log(FaceLandmarker.FACE_LANDMARKS_LEFT_EYE);
          console.log("FACE_LANDMARKS_RIGHT_EYEBROW");
          console.log(FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW);
          console.log("FACE_LANDMARKS_LEFT_EYEBROW");
          console.log(FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW);
/*
          drawingUtils.drawConnectors(
            landmarks,
            FaceLandmarker.FACE_LANDMARKS_LIPS,
            { color: "#0000FF", width: "1px" } // Lips color (#E0E0E0 is default)
          );
        */
        }
      } else {
        faceOutput.innerText = "No face landmarks detected.";
      }

      if (handGestureRunning) {
        const nowInMs = Date.now();
        const handResults = await gestureRecognizer.recognizeForVideo(video, nowInMs);

        canvasCtx.save();

        if (handResults.landmarks.length > 0) {
          const drawingUtils = new DrawingUtils(canvasCtx);
          let handIndex = 0;
          for (const landmarks of handResults.landmarks) {
            drawingUtils.drawConnectors(
              landmarks,
              GestureRecognizer.HAND_CONNECTIONS,
              { color: "#7696eb", lineWidth: 5 } // Landmark connection lines (default 00FF00)
            );
            // 21 landmark points
            drawingUtils.drawLandmarks(landmarks, { color: "#22dee5", lineWidth: 2 }); // #FF0000

            const gestures = handResults.gestures[handIndex];
            const handedness = handResults.handednesses[handIndex];
            if (gestures && gestures.length > 0) {
              const gestureName = gestures[0].categoryName;
              //gestureOutput.innerText = gesture_name_map[gestureName] || "Unknown Gesture";
              gestureOutput.innerText = gestureNameMap[gestureName] || "Unknown Gesture";
              //gestureOutput.innerText = `${gestures[0].categoryName}`;
              confidenceOutput.innerText = `${(gestures[0].score * 100).toFixed(2)}%`;
              handednessOutput.innerText = `${handedness[0].categoryName}`;
              sendGestureToServer(gestureName); // Send gesture to server
            } else {
              gestureOutput.innerText = "Not Detected";
              confidenceOutput.innerText = "100%";
              handednessOutput.innerText = "Not Detected";
            }
            handIndex++;
          }
        } else {
          gestureOutput.innerText = "Not Detected";
          confidenceOutput.innerText = "100%";
          handednessOutput.innerText = "Not Detected";
        }

        handCountOutput.innerText = `${handResults.landmarks.length}`;

        canvasCtx.restore();
      }

      window.requestAnimationFrame(predictWebcam);
    }
  }