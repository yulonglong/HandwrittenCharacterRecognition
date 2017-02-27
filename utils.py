import errno
import logging
import os
import re
import subprocess as sub
import sys


#-----------------------------------------------------------------------------------------------------------#

def set_logger(out_dir=None):
	console_format = BColors.OKBLUE + '[%(levelname)s]' + BColors.ENDC + ' (%(name)s) (%(asctime)s) %(message)s'
	#datefmt='%Y-%m-%d %Hh-%Mm-%Ss'
	logger = logging.getLogger()
	logger.setLevel(logging.DEBUG)
	console = logging.StreamHandler()
	console.setLevel(logging.DEBUG)
	console.setFormatter(logging.Formatter(console_format, datefmt='%d/%m %I:%M:%S %p'))
	logger.addHandler(console)
	if out_dir:
		file_format = '[%(levelname)s] (%(name)s) (%(asctime)s) %(message)s'
		log_file = logging.FileHandler(out_dir + '/log.txt', mode='w')
		log_file.setLevel(logging.DEBUG)
		log_file.setFormatter(logging.Formatter(file_format, datefmt='%d/%m %I:%M:%S %p'))
		logger.addHandler(log_file)

#-----------------------------------------------------------------------------------------------------------#

def __shell(command):
	return sub.Popen(command, shell=True, stdout=sub.PIPE, stderr=sub.PIPE)
	
# Currently the best
def capture(command):
	out, err, code = capture_all(command)
	assert (code == 0), "Failed to run the command: " + command
	return out

# Good, if more info is needed
def capture_all(command):
	p = __shell(command)
	output, err = p.communicate()
	return output, err, p.returncode
	
# Better to avoid
def capture_no_assert(command):
	p = __shell(command)
	return p.stdout.read()
	
# Not well-tested, but should be good
def capture_output(command):
	try:
		eval("sub.check_output")
	except:
		error("subprocess check_output function is not supported in this python version:" + version())
	output = sub.check_output(command, shell=True)
	return output

def mkdir_p(path):
	if path == '':
		return
	try:
		os.makedirs(path)
	except OSError as exc: # Python >2.5
		if exc.errno == errno.EEXIST and os.path.isdir(path):
			pass
		else: raise

def get_root_dir():
	return os.path.dirname(sys.argv[0])

#-----------------------------------------------------------------------------------------------------------#
	

class BColors:
	HEADER = '\033[95m'
	OKBLUE = '\033[94m'
	OKGREEN = '\033[92m'
	WARNING = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'
	WHITE = '\033[37m'
	YELLOW = '\033[33m'
	GREEN = '\033[32m'
	BLUE = '\033[34m'
	CYAN = '\033[36m'
	RED = '\033[31m'
	MAGENTA = '\033[35m'
	BLACK = '\033[30m'
	BHEADER = BOLD + '\033[95m'
	BOKBLUE = BOLD + '\033[94m'
	BOKGREEN = BOLD + '\033[92m'
	BWARNING = BOLD + '\033[93m'
	BFAIL = BOLD + '\033[91m'
	BUNDERLINE = BOLD + '\033[4m'
	BWHITE = BOLD + '\033[37m'
	BYELLOW = BOLD + '\033[33m'
	BGREEN = BOLD + '\033[32m'
	BBLUE = BOLD + '\033[34m'
	BCYAN = BOLD + '\033[36m'
	BRED = BOLD + '\033[31m'
	BMAGENTA = BOLD + '\033[35m'
	BBLACK = BOLD + '\033[30m'
	
	@staticmethod
	def cleared(s):
		return re.sub("\033\[[0-9][0-9]?m", "", s)

def red(message):
	return BColors.RED + str(message) + BColors.ENDC

def b_red(message):
	return BColors.BRED + str(message) + BColors.ENDC

def blue(message):
	return BColors.BLUE + str(message) + BColors.ENDC

def b_yellow(message):
	return BColors.BYELLOW + str(message) + BColors.ENDC

def green(message):
	return BColors.GREEN + str(message) + BColors.ENDC

def b_green(message):
	return BColors.BGREEN + str(message) + BColors.ENDC
	
#-----------------------------------------------------------------------------------------------------------#

def is_gpu_free(gpu_id):
	out = capture('nvidia-smi -i ' + str(gpu_id)).strip()
	tokens = out.split('\n')[-2].split()
	return ' '.join(tokens[1:5]) == 'No running processes found'

def set_theano_device(device, threads):
	logger = logging.getLogger(__name__)
	assert device == "cpu" or device.startswith("gpu"), "The device can only be 'cpu', 'gpu' or 'gpu<id>'"
	if device.startswith("gpu") and len(device) > 3:
		try:
			gpu_id = int(device[3:])
			if not is_gpu_free(gpu_id):
				logger.warning('The selected GPU (GPU' + str(gpu_id) + ') is apparently busy.')
		except ValueError:
			logger.error("Unknown GPU device format: " + device)
			sys.exit()
	if device.startswith("gpu"):
		logger.warning('Running on GPU yields non-deterministic results.')
	assert sys.modules.has_key('theano') == False, "nmt.utils.set_theano_device() function cannot be called after importing theano"
	os.environ['OMP_NUM_THREADS'] = str(threads)
	os.environ['THEANO_FLAGS'] = 'device=' + device
	os.environ['THEANO_FLAGS'] += ',force_device=True'
	os.environ['THEANO_FLAGS'] += ',floatX=float32'
	#os.environ['THEANO_FLAGS'] += ',warn_float64=warn'
	#os.environ['THEANO_FLAGS'] += ',cast_policy=numpy+floatX'
	#os.environ['THEANO_FLAGS'] += ',allow_gc=True'
	os.environ['THEANO_FLAGS'] += ',print_active_device=False'
	#os.environ['THEANO_FLAGS'] += ',mode=FAST_COMPILE'
	os.environ['THEANO_FLAGS'] += ',mode=FAST_RUN'
	os.environ['THEANO_FLAGS'] += ',on_unused_input=warn'
	#os.environ['THEANO_FLAGS'] += ',nvcc.fastmath=True' # makes div and sqrt faster at the cost of precision
	os.environ['THEANO_FLAGS'] += ',optimizer_including=cudnn' # Comment out if CUDNN is not available
	import theano
	if theano.config.device == "gpu":
		logger.info(
			"Device: " + theano.config.device.upper() + " "
			+ str(theano.sandbox.cuda.active_device_number())
			+ " (" + str(theano.sandbox.cuda.active_device_name()) + ")"
		)
	else:
		logger.info("Device: " + theano.config.device.upper())

#-----------------------------------------------------------------------------------------------------------#

def print_args(args, path=None):
	if path:
		output_file = open(path, 'w')
	logger = logging.getLogger(__name__)
	logger.info("Arguments:")
	args.command = ' '.join(sys.argv)
	items = vars(args)
	for key in sorted(items.keys(), key=lambda s: s.lower()):
		value = items[key]
		if not value:
			value = "None"
		logger.info("  " + key + ": " + str(items[key]))
		if path is not None:
			output_file.write("  " + key + ": " + str(items[key]) + "\n")
	if path:
		output_file.close()
	del args.command
