<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=iso-8859-1" />
<title>Untitled Document</title>
</head>

<body>
<?php
//$file="http://iotcloud.co.in/testsms/atm_capture/img.txt";
//$fp=fopen($file,"r");
//$r=fread($fp,filesize($file));
//echo $r;

$dd=file_get_contents("http://iotcloud.co.in/testsms/atm_capture/log.txt");
$fp=fopen("log.txt","w");
fwrite($fp,$dd);

?>
</body>
</html>
