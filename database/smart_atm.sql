-- phpMyAdmin SQL Dump
-- version 2.11.6
-- http://www.phpmyadmin.net
--
-- Host: localhost
-- Generation Time: Dec 15, 2021 at 05:42 PM
-- Server version: 5.0.51
-- PHP Version: 5.2.6

SET SQL_MODE="NO_AUTO_VALUE_ON_ZERO";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;

--
-- Database: `smart_atm`
--

-- --------------------------------------------------------

--
-- Table structure for table `admin`
--

CREATE TABLE `admin` (
  `username` varchar(20) NOT NULL,
  `password` varchar(20) NOT NULL,
  `amount` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `admin`
--

INSERT INTO `admin` (`username`, `password`, `amount`) VALUES
('admin', 'admin', 49500);

-- --------------------------------------------------------

--
-- Table structure for table `event`
--

CREATE TABLE `event` (
  `id` int(11) NOT NULL,
  `name` varchar(50) NOT NULL,
  `accno` int(11) NOT NULL,
  `amount` int(11) NOT NULL,
  `rdate` varchar(50) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `event`
--

INSERT INTO `event` (`id`, `name`, `accno`, `amount`, `rdate`) VALUES
(1, 'Suresh', 2147483647, 500, '15-12-2021'),
(2, 'Suresh', 2147483647, 500, '15-12-2021'),
(3, 'Suresh', 2147483647, 500, '15-12-2021');

-- --------------------------------------------------------

--
-- Table structure for table `numbers`
--

CREATE TABLE `numbers` (
  `id` int(11) NOT NULL,
  `number` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `numbers`
--

INSERT INTO `numbers` (`id`, `number`) VALUES
(1, 0),
(2, 1),
(3, 2),
(4, 3),
(5, 4),
(6, 5),
(7, 6),
(8, 7),
(9, 8),
(10, 9);

-- --------------------------------------------------------

--
-- Table structure for table `register`
--

CREATE TABLE `register` (
  `id` int(11) NOT NULL,
  `name` varchar(20) NOT NULL,
  `address` varchar(200) NOT NULL,
  `mobile` bigint(20) NOT NULL,
  `email` varchar(50) NOT NULL,
  `accno` varchar(20) NOT NULL,
  `card` varchar(20) NOT NULL,
  `bank` varchar(20) NOT NULL,
  `branch` varchar(20) NOT NULL,
  `deposit` int(11) NOT NULL,
  `username` varchar(20) NOT NULL,
  `password` varchar(20) NOT NULL,
  `rdate` varchar(20) NOT NULL,
  `aadhar1` varchar(20) NOT NULL,
  `aadhar2` varchar(20) NOT NULL,
  `aadhar3` varchar(20) NOT NULL,
  `face_st` int(11) NOT NULL,
  `fimg` varchar(30) NOT NULL,
  `otp` varchar(20) NOT NULL,
  `allow_st` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `register`
--

INSERT INTO `register` (`id`, `name`, `address`, `mobile`, `email`, `accno`, `card`, `bank`, `branch`, `deposit`, `username`, `password`, `rdate`, `aadhar1`, `aadhar2`, `aadhar3`, `face_st`, `fimg`, `otp`, `allow_st`) VALUES
(1, 'Suresh', '22,FG Nagar, Trichy', 9080891715, 'suresh@gmail.com', '2233440001', '564500015155', 'SBI', 'Trichy', 4500, '', '1234', '17-10-2021', '432365447121', '', '', 0, '1_25.jpg', '', 0),
(2, 'Dharun', 'DS Nagar, Salem', 7402333204, 'dharun@gmail.com', '2233440002', '381700023499', 'SBI', 'Salem', 10000, '', '', '17-10-2021', '532365447122', '', '', 0, '', '', 0),
(3, 'Siva', 'SR Nagar,Madurai', 9500158357, 'siva@gmail.com', '2233440003', '340000035849', 'SBI', 'Madurai', 10000, '', '2213', '17-10-2021', '347375447342', '', '', 0, '3_23.jpg', '', 0),
(4, 'Raj', '22,FG Nagar, Trichy', 8870136390, 'rndittrichy@gmail.com', '2233440004', '372900041243', 'SBI', 'Trichy', 10000, '', '6288', '24-10-2021', '456453554353', '', '', 0, '4_15.jpg', '', 0),
(5, 'Mani', 'kk nagar', 9600721896, 'mani.c@rndit.co.in', '2233440005', '376500052944', 'SBI', 'Trichy', 10000, '', '9512', '14-12-2021', '347781339766', '', '', 0, '5_14.jpg', '', 0);

-- --------------------------------------------------------

--
-- Table structure for table `vt_face`
--

CREATE TABLE `vt_face` (
  `id` int(11) NOT NULL,
  `vid` int(11) NOT NULL,
  `vface` varchar(30) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `vt_face`
--

INSERT INTO `vt_face` (`id`, `vid`, `vface`) VALUES
(3, 5, '5_2.jpg'),
(4, 5, '5_3.jpg'),
(5, 5, '5_4.jpg'),
(6, 5, '5_5.jpg'),
(7, 5, '5_6.jpg'),
(8, 5, '5_7.jpg'),
(9, 5, '5_8.jpg'),
(10, 5, '5_9.jpg'),
(11, 5, '5_10.jpg'),
(12, 5, '5_11.jpg'),
(13, 5, '5_12.jpg'),
(14, 5, '5_13.jpg'),
(15, 5, '5_14.jpg'),
(16, 1, '1_2.jpg'),
(17, 1, '1_3.jpg'),
(18, 1, '1_4.jpg'),
(19, 1, '1_5.jpg'),
(20, 1, '1_6.jpg'),
(21, 1, '1_7.jpg'),
(22, 1, '1_8.jpg'),
(23, 1, '1_9.jpg'),
(24, 1, '1_10.jpg'),
(25, 1, '1_11.jpg'),
(26, 1, '1_12.jpg'),
(27, 1, '1_13.jpg'),
(28, 1, '1_14.jpg'),
(29, 1, '1_15.jpg'),
(30, 1, '1_16.jpg'),
(31, 1, '1_17.jpg'),
(32, 1, '1_18.jpg'),
(33, 1, '1_19.jpg'),
(34, 1, '1_20.jpg'),
(35, 1, '1_21.jpg'),
(36, 1, '1_22.jpg'),
(37, 1, '1_23.jpg'),
(38, 1, '1_24.jpg'),
(39, 1, '1_25.jpg');
