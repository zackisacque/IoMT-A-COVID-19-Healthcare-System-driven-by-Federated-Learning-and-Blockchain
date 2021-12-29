'''
title           : FedMedChain.py
description     : Towards an Efficient IoMT: Privacy Infrastructure for COVID-19 Pandemic based on Federated Learning and Blockchain Technology
authors         : Omaji Samuel, Akogwu Blessing Omojo, Abdulkarin Musa Onuja, Yunisa Sunday, Prayag Tiwari, Deepak
                  Gupta, Ghulam Hafeez, Adamu Sani Yahaya, Oluwaseun Jumoke Fatoba, and Shahab Shamshirband
date_created    : 20211112
date_modified   : Not Applicable
version         : 0.1
usage           : python FedMedChain.py
                  python FedMedChain.py -p 5000
                  python FedMedChain.py --port 5000
python_version  : 3.7.9
Comments        : The consortium blockchain implementation based on [1,2]
                  We made modifications in the proof of work of [1] to select nodes using a proposed two rounds reinforcing addition game
                  We also proposed the federated learning (FL) method and epidemiological model for COVID-19.
References      : [1] https://github.com/dvf/blockchain/blob/master/blockchain.py
                  [2] https://github.com/julienr/ipynb_playground/blob/master/bitcoin/dumbcoin/dumbcoin.ipynb
'''

from collections import OrderedDict

import binascii

import Crypto
import Crypto.Random
from Crypto.Hash import SHA
from Crypto.PublicKey import RSA
from Crypto.Signature import PKCS1_v1_5
from ReinforcingAdditionGame import ConsensusProtocol
import hashlib
import json
import time
from urllib.parse import urlparse
from uuid import uuid4
from BlockchainbasedFederatedLearning import *
import requests
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS

from matplotlib import pyplot as P

MINING_SENDER = "THE BLOCKCHAIN"
MINING_REWARD = 1
MINING_DIFFICULTY = 2  # 4 billion in range(32)


class Blockchain:
    # initialization of parameters
    def __init__(self):
        
        self.transactions = []
        self.chain = [] # generate the chain
        self.nodes = set() # set the nodes of the blockchain
        #Generate random number to be used as node_id, the number will also be used for the two rounds reinforcing game
        self.node_id = str(uuid4()).replace('-', '') 
        #Create genesis block
        self.create_block(0, '00') # create the initial block
        self.Ro = [] # Ro is the reproduction number of the epidemiological model

    def register_node(self, node_url):
        """
        Add a new node to the list of nodes
        """
        #Checking node_url has valid format
        parsed_url = urlparse(node_url)
        if parsed_url.netloc:
            self.nodes.add(parsed_url.netloc)
        elif parsed_url.path:
            # Accepts an URL without scheme like '192.168.0.5:5000'.
            self.nodes.add(parsed_url.path)
        else:
            raise ValueError('Invalid URL')


    def verify_transaction_signature(self, sender_address, signature, transaction):
        """
        Check that the provided signature corresponds to transaction
        signed by the public key (sender_address)
        """
        public_key = RSA.importKey(binascii.unhexlify(sender_address))
        verifier = PKCS1_v1_5.new(public_key)
        h = SHA.new(str(transaction).encode('utf8'))
        return verifier.verify(h, binascii.unhexlify(signature))


    def submit_transaction(self, sender_address, recipient_address, value, signature):
        """
        Add a transaction to transactions array if the signature verified
        """
        transaction = OrderedDict({'sender_address': sender_address, 
                                    'recipient_address': recipient_address,
                                    'value': value})

        #Reward for mining a block
        if sender_address == request.form['sender_address']:
            self.transactions.append(transaction)
            return len(self.chain) + 1
        #Manages transactions from wallet to another wallet
        else:
            transaction_verification = self.verify_transaction_signature(sender_address, signature, transaction)
            if transaction_verification:
                self.transactions.append(transaction)
                return len(self.chain) + 1
            else:
                return False


    def create_block(self, nonce, previous_hash):
        """
        Add a block of transactions to the blockchain
        """
        block = {'block_number': len(self.chain) + 1,
                'timestamp': time.time(),
                'transactions': self.transactions,
                'nonce': nonce,
                'previous_hash': previous_hash}

        # Reset the current list of transactions
        self.transactions = []

        self.chain.append(block)
        return block


    def hash(self, block):
        """
        Create a SHA-256 hash of a block
        """
        # We must make sure that the Dictionary is Ordered, or we'll have inconsistent hashes
        block_string = json.dumps(block, sort_keys=True).encode()
        
        return hashlib.sha256(block_string).hexdigest()


    def proof_of_COvidCoin(self):
        """
        Proof of COvidCoin algorithm
        """
        start_time = time.time()
        last_block = self.chain[-1]
        last_hash = self.hash(last_block)

        nonce = 0
        while self.valid_proof(self.transactions, last_hash, nonce) is False:
            nonce += 1
            
        end_time = time.time()
        elapsed_time = end_time - start_time
        print ("Elapsed time: %.4f seconds" % elapsed_time)

        if elapsed_time > 0: 

            hash_power = float(int(nonce)/elapsed_time)
            print ("Hashing power: %ld hashes per second" % hash_power)
        return nonce


    def valid_proof(self, transactions, last_hash, nonce, difficulty=MINING_DIFFICULTY):
        """
        Check if a hash value satisfies the mining conditions. This function is used within the proof_of_COvidCoin function.
        """
        target = (2 ** (256-difficulty))* ConsensusProtocol.TwoRoundsAdditionGame(10)
        guess = (str(transactions)+str(last_hash)+str(nonce)).encode('utf8')
        hash_result = hashlib.sha256(guess).hexdigest()
        if int(hash_result, 16) < target:
            print ("Success with nonce %d" % nonce)
            print ("Hash is %s" % hash_result)
            return hash_result[:difficulty] == '0'*difficulty
        print("Failed after %d (max_nonce) tries " % nonce)
        
        return nonce


    def valid_chain(self, chain):
        """
        check if a bockchain is valid
        """
        last_block = chain[0]
        current_index = 1

        while current_index < len(chain):
            block = chain[current_index]
            print(last_block)
            print(block)
            #print("\n-----------\n")
            # Check that the hash of the block is correct
            if block['previous_hash'] != self.hash(last_block):
                return False

            # Check that the Proof of Work is correct
            #Delete the reward transaction
            transactions = block['transactions'][:-1]
            # Need to make sure that the dictionary is ordered. Otherwise we'll get a different hash
            transaction_elements = ['sender_address', 'recipient_address', 'value']
            transactions = [OrderedDict((k, transaction[k]) for k in transaction_elements) for transaction in transactions]

            if not self.valid_proof(transactions, block['previous_hash'], block['nonce'], MINING_DIFFICULTY):
                return False

            last_block = block
            current_index += 1
        return True

    def resolve_conflicts(self):
        """
        Resolve conflicts between blockchain's nodes
        by replacing our chain with the longest one in the network.
        """
        neighbours = self.nodes
        new_chain = None

        # We're only looking for chains longer than ours
        max_length = len(self.chain)

        # Grab and verify the chains from all the nodes in our network
        for node in neighbours:
            print('http://' + node + '/chain')
            response = requests.get('http://' + node + '/chain')

            if response.status_code == 200:
                length = response.json()['length']
                chain = response.json()['chain']

                # Check if the length is longer and the chain is valid
                if length > max_length and self.valid_chain(chain):
                    max_length = length
                    new_chain = chain

        # Replace our chain if we discovered a new, valid chain longer than ours
        if new_chain:
            self.chain = new_chain
            return True

        return False
# Create class for client transaction
class Transaction:

    def __init__(self, sender_address, sender_private_key, recipient_address, value):
        self.sender_address = sender_address
        self.sender_private_key = sender_private_key
        self.recipient_address = recipient_address
        self.value = value

    def __getattr__(self, attr):
        return self.data[attr]

    def to_dict(self):
        return OrderedDict({'sender_address': self.sender_address,
                            'recipient_address': self.recipient_address,
                            'value': self.value})

    def sign_transaction(self):
        """
        Sign transaction with private key
        """
        private_key = RSA.importKey(binascii.unhexlify(self.sender_private_key))
        signer = PKCS1_v1_5.new(private_key)
        h = SHA.new(str(self.to_dict()).encode('utf8'))
        return binascii.hexlify(signer.sign(h)).decode('ascii')

# Instantiate the Node
app = Flask(__name__)
CORS(app)

# Instantiate the Blockchain
blockchain = Blockchain()
# Perform the federated learning process
print("Performing the blockchain based federated learning process...")
# configuration of federated learning
config = {
        'n_CDCs': 5, # number of CDCs
        'key_length': 1024, # length of encryption key
        'n_iter': 50, # number of iterations
        'eta': 0.01, # learning rate
    }
# load data, train/test split and split training data between CDCs
X, y, X_test, y_test = get_data(n_CDCs=config['n_CDCs'])
# first each CDC learns a model on its respective dataset for comparison.
with timer() as t:
    local_learning(X, y, X_test, y_test, config)
# and now the full glory of federated learning
with timer() as t:
    blockchain_based_federated_learning(X, y, X_test, y_test, config)
@app.route('/')
def index():
    return render_template('./index.html')

@app.route('/configure')
def configure():
    return render_template('./configure.html')

@app.route('/transactions/new', methods=['POST'])
def new_transaction():
    values = request.form

    # Check that the required fields are in the POST'ed data
    required = ['sender_address', 'recipient_address', 'amount', 'signature']
    if not all(k in values for k in required):
        return 'Missing values', 400
    # Create a new Transaction
    transaction_result = blockchain.submit_transaction(values['sender_address'], values['recipient_address'], values['amount'], values['signature'])

    if transaction_result == False:
        response = {'message': 'Invalid Transaction!'}
        return jsonify(response), 406
    else:
        response = {'message': 'Transaction will be added to Block '+ str(transaction_result)}
        return jsonify(response), 201

@app.route('/transactions/get', methods=['GET'])
def get_transactions():
    #Get transactions from transactions pool
    transactions = blockchain.transactions

    response = {'transactions': transactions}
    return jsonify(response), 200

@app.route('/chain', methods=['GET'])
def full_chain():
    response = {
        'chain': blockchain.chain,
        'length': len(blockchain.chain),
    }
    return jsonify(response), 200

@app.route('/mine', methods=['GET'])
def mine():
    # We run the proof of COVIDCoin algorithm to get the next proof...
    last_block = blockchain.chain[-1]
    nonce = blockchain.proof_of_COvidCoin()

    # We must receive a reward for finding the proof.
    blockchain.submit_transaction(sender_address = request.form['sender_address'], recipient_address = blockchain.node_id, value=MINING_REWARD, signature="")

    # Forge the new Block by adding it to the chain
    previous_hash = blockchain.hash(last_block)
    block = blockchain.create_block(nonce, previous_hash)

    response = {
        'message': "New Block Forged",
        'block_number': block['block_number'],
        'transactions': block['transactions'],
        'nonce': block['nonce'],
        'previous_hash': block['previous_hash'],
    }
    return jsonify(response), 200


@app.route('/nodes/register', methods=['POST'])
def register_nodes():
    values = request.form
    nodes = values.get('nodes').replace(" ", "").split(',')

    if nodes is None:
        return "Error: Please supply a valid list of nodes", 400

    for node in nodes:
        blockchain.register_node(node)

    response = {
        'message': 'New nodes have been added',
        'total_nodes': [node for node in blockchain.nodes],
    }
    return jsonify(response), 201


@app.route('/nodes/resolve', methods=['GET'])
def consensus():
    replaced = blockchain.resolve_conflicts()

    if replaced:
        response = {
            'message': 'Our chain was replaced',
            'new_chain': blockchain.chain
        }
    else:
        response = {
            'message': 'Our chain is authoritative',
            'chain': blockchain.chain
        }
    return jsonify(response), 200


@app.route('/nodes/get', methods=['GET'])
def get_nodes():
    nodes = list(blockchain.nodes)
    response = {'nodes': nodes}
    return jsonify(response), 200

# make the client end transaction
@app.route('/make/transaction')
def make_transaction():
    return render_template('./make_transaction.html')

@app.route('/view/transactions')
def view_transaction():
    return render_template('./view_transactions.html')

@app.route('/wallet/new', methods=['GET'])
def new_wallet():
    random_gen = Crypto.Random.new().read
    private_key = RSA.generate(1024, random_gen)
    public_key = private_key.publickey()
    response = {
        'private_key': binascii.hexlify(private_key.exportKey(format='DER')).decode('ascii'),
        'public_key': binascii.hexlify(public_key.exportKey(format='DER')).decode('ascii')
    }

    return jsonify(response), 200

@app.route('/generate/transaction', methods=['POST'])
def generate_transaction():
    
    sender_address = request.form['sender_address']
    sender_private_key = request.form['sender_private_key']
    recipient_address = request.form['recipient_address']
    value = request.form['amount']

    transaction = Transaction(sender_address, sender_private_key, recipient_address, value)

    response = {'transaction': transaction.to_dict(), 'signature': transaction.sign_transaction()}

    return jsonify(response), 200

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-p', '--port', default=5000, type=int, help='port to listen on')
    args = parser.parse_args()
    port = args.port

    app.run(host='127.0.0.1', port=port)

    
    r=3
    R=5
    c= 0.4
    a=6
    b=0.6
    e=0.1
    k=0.1
    N=round(user_choice(r,c,a,b,e))
    print(N)
    TC=aggregator_total_cost(k,r,N) 
    print(TC)
    R_agg=aggregator_revenue(r,N) 
    print(R_agg)
    U_agg=aggregator_utility(R,r,k,N)
    print(U_agg)
        
    
    plt.figure(figsize=(10, 7))
    y_pos= np.arange(len(TC))
    performance=TC
    Objects=range(N)
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, Objects)
    plt.ylabel('Total Cost')
    plt.xlabel('Number of Nodes')
    plt.show()
    # plots the graphs for federated learning and local learning
    # plotting the points  
    
    x1=[1,2,3,4,5]
    plt.plot(x1, mse1,label='federated learning',color='green', linestyle='dashed', linewidth = 3, 
         marker='o', markerfacecolor='blue', markersize=12); 
    
    plt.plot(x1, mse2,label='local learning',color='blue', linestyle='dashed', linewidth = 3, 
        marker='o', markerfacecolor='red', markersize=12) 
  
    # naming the x axis 
    plt.xlabel('Homes') 
    # naming the y axis 
    plt.ylabel('Error') 
  
    # giving a title to my graph 
    plt.title('federated learning versus local learning') 
    # show a legend on the plot 
    plt.legend() 
  
    # function to show the plot 
    plt.show() 






