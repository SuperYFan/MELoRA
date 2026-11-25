import errno
import os
    
def assert_exits(path):
    assert os.path.exists(path), 'Does not exist : {}'.format(path)
    
class Logger(object):
    def __init__(self,output_dir=None, mode='w'):
        self.log_file=None
        if output_dir:
            dirname=os.path.dirname(output_dir)
            if dirname and not os.path.exists(dirname):
                os.makedirs(dirname, exist_ok=True)
            self.log_file=open(output_dir,mode)
        self.infos={}
        
    def append(self,key,val):
        vals=self.infos.setdefault(key,[])
        vals.append(val)

    def log(self,extra_msg=''):
        msgs=[extra_msg]
        for key, vals in self.infos.iteritems():
            msgs.append('%s %.6f' %(key,np.mean(vals)))
        msg='\n'.joint(msgs)
        self.log_file.write(msg+'\n')
        self.log_file.flush()
        self.infos={}
        return msg
        
    def write(self,msg):
        if self.log_file is not None:
            self.log_file.write(msg+'\n')
            self.log_file.flush()
        print(msg)    
