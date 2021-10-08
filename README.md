# manthan.-abstraction-based-summarization
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# PyInstaller
#  Usually these files are written by a python script from a template
#  before PyInstaller builds the exe, so as to inject date/other infos into it.
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# pyenv
.python-version

# celery beat schedule file
celerybeat-schedule

# SageMath parsed files
*.sage.py

# dotenv
.env

# virtualenv
.venv
venv/
ENV/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/

# forward encoding
S = tf.shape(embd_text)[1] #text sequence length
N = tf.shape(embd_text)[0] #batch_size

i=0
hidden=tf.zeros([N, hidden_size], dtype=tf.float32)
cell=tf.zeros([N, hidden_size], dtype=tf.float32)
hidden_forward=tf.TensorArray(size=S, dtype=tf.float32)

#shape of embd_text: [N,S,embd_dim]
embd_text_t = tf.transpose(embd_text,[1,0,2]) 
#current shape of embd_text: [S,N,embd_dim]

def cond(i, hidden, cell, hidden_forward):
    return i < S

def body(i, hidden, cell, hidden_forward):
    x = embd_text_t[i]
    
    hidden,cell = LSTM(x,hidden,cell,embd_dim,hidden_size,scope="forward_encoder")
    hidden_forward = hidden_forward.write(i, hidden)

    return i+1, hidden, cell, hidden_forward

_, _, _, hidden_forward = tf.while_loop(cond, body, [i, hidden, cell, hidden_forward])

# backward encoding
i=S-1
hidden=tf.zeros([N, hidden_size], dtype=tf.float32)
cell=tf.zeros([N, hidden_size], dtype=tf.float32)
hidden_backward=tf.TensorArray(size=S, dtype=tf.float32)

def cond(i, hidden, cell, hidden_backward):
    return i >= 0

def body(i, hidden, cell, hidden_backward):
    x = embd_text_t[i]
    hidden,cell = LSTM(x,hidden,cell,embd_dim,hidden_size,scope="backward_encoder")
    hidden_backward = hidden_backward.write(i, hidden)

    return i-1, hidden, cell, hidden_backward

_, _, _, hidden_backward = tf.while_loop(cond, body, [i, hidden, cell, hidden_backward])

# merge
hidden_forward = hidden_forward.stack()
hidden_backward = hidden_backward.stack()

encoder_states = tf.concat([hidden_forward,hidden_backward],axis=-1)
encoder_states = tf.transpose(encoder_states,[1,0,2])

encoder_states = dropout(encoder_states,rate=0.3,training=tf_train)

final_encoded_state = dropout(tf.concat([hidden_forward[-1],hidden_backward[-1]],axis=-1),rate=0.3,training=tf_train)


# Implementation of attention scoring function
def attention_score(encoder_states,decoder_hidden_state,scope="attention_score"):
    
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        Wa = tf.get_variable("Wa", shape=[2*hidden_size,2*hidden_size],
                                    dtype=tf.float32,
                                    trainable=True,
                                    initializer=tf.glorot_uniform_initializer())
        
    encoder_states = tf.reshape(encoder_states,[N*S,2*hidden_size])
    
    encoder_states = tf.reshape(tf.matmul(encoder_states,Wa),[N,S,2*hidden_size])
    decoder_hidden_state = tf.reshape(decoder_hidden_state,[N,2*hidden_size,1])
    
    return tf.reshape(tf.matmul(encoder_states,decoder_hidden_state),[N,S])

# Embed vectorized text
embd_text = tf.nn.embedding_lookup(tf_embd, tf_text)

embd_text = dropout(embd_text,rate=0.3,training=tf_train)


