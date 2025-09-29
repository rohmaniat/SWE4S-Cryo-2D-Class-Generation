# Let's call this file func_testing.sh until we have another functional test file. Then let's call them something more specific.

test -e ssshtest || curl -q -O https://raw.githubusercontent.com/ryanlayer/ssshtest/master/ssshtest
. ssshtest

