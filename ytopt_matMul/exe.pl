#!/usr/bin/env perl

#Author: Xingfu Wu
#MCS, ANL
# exe.pl: average the execution time in 5 runs
#Modified: Praveen Paramasivam
use Time::HiRes qw(gettimeofday); 

$A_FILE = "tmpfile.txt";
my $acc = -1;
foreach $filename (@ARGV) {
 #  print "Start to preprocess ", $filename, "...\n";
    system("python $filename > tmpfile.txt");
    open (TEMFILE, '<', $A_FILE);
    while (<TEMFILE>) {
        $line = $_;
        chomp ($line);

        if ($line =~ /Elpased time/) {
                ($v1,$v2)= split('\=', $line);
		 $acc = $v2;
        }
   }
   close(TEMFILE);
   if ($acc) {
	printf("%.3f", $acc)
   }
 #  system("unlink  tmpfile.txt");
}
