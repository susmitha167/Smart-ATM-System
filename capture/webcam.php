<!doctype html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>How to capture picture from webcam with Webcam.js</title>

</head>
<body>

	<!-- CSS -->
    <style>
    #my_camera{
        width: 400px;
        height: 400px;
        /*border: 1px solid black;*/
    }
	</style>

	<!-- -->
	<div id="my_camera"></div>
	
	
  <div id="results"  ></div>
	
	<!-- Script -->
	<script type="text/javascript" src="webcamjs/webcam.min.js"></script>

	<!-- Code to handle taking the snapshot and displaying it locally -->
	<script language="JavaScript">
		
		// Configure a few settings and attach camera
		
			Webcam.set({
				width: 400,
				height: 400,
				image_format: 'jpeg',
				jpeg_quality: 90
			});
			Webcam.attach( '#my_camera' );
		
		// A button for taking snaps
		

		// preload shutter audio clip
		var shutter = new Audio();
		shutter.autoplay = false;
		//shutter.src = navigator.userAgent.match(/Firefox/) ? 'shutter.ogg' : 'shutter.mp3';

		function take_snapshot() {
			// play sound effect
			shutter.play();

			// take snapshot and get image data
			Webcam.snap( function(data_uri) {
				// display results in page
				document.getElementById('results').innerHTML = 
					'<img id="imageprev" src="'+data_uri+'"/>';
			} );

			Webcam.reset();
			saveSnap();
		}

		function saveSnap(){
			// Get base64 value from <img id='imageprev'> source
			var base64image =  document.getElementById("imageprev").src;

			 Webcam.upload( base64image, 'upload.php', function(code, text) {
				 console.log('Save successfully');
				 //console.log(text);
            });

		}
		
		setTimeout(function () {
   //Redirect with JavaScript
  take_snapshot();
  
}, 5000);
	</script>
	
	
	
	
	
	<script language="javascript">
	setTimeout(function () {
   //Redirect with JavaScript
 window.location.href="webcapture.html";
  
}, 15000);
	</script>
</body>
</html>
