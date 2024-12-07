<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=iso-8859-1" />
<title>Untitled Document</title>
</head>

<body>
<?php
//$file="upload/myfile.txt";
////"http://localhost/sample/f1.jpg"
//$current = file_get_contents("http://localhost/sample/myfile.txt");
//file_put_contents($file,$current);

/*$content = file_get_contents("http://localhost/sample/f1.jpg");
//Store in the filesystem.
$fp = fopen("ss.jpg", "w");
fwrite($fp, $content);
fclose($fp);*/

/////////////////////////////////////////////////////////////////////
// Remote image URL
/*$url = 'http://localhost/sample/f1.jpg';

// Image path
$img = 'aa.jpg';

// Save image
$ch = curl_init($url);
$fp = fopen($img, 'wb');
curl_setopt($ch, CURLOPT_FILE, $fp);
curl_setopt($ch, CURLOPT_HEADER, 0);
curl_exec($ch);
curl_close($ch);
fclose($fp);*/
/////////////////////////////////////////
//copy('http://localhost/sample/f1.jpg', 'https://iotcloud.co.in/testsms/cc.jpg');


//extract($_REQUEST);
//move_uploaded_file();
<?php
//upload.php
if($_FILES["file"]["name"] != '')
{
 $test = explode('.', $_FILES["file"]["name"]);
 $ext = end($test);
 $name = rand(100, 999) . '.' . $ext;
 $location = './upload/' . $name;  
 move_uploaded_file($_FILES["file"]["tmp_name"], $location);
 echo '<img src="'.$location.'" height="150" width="225" class="img-thumbnail" />';
}
?>
?>

</body>
</html>
