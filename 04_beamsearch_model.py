
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf


# In[2]:

TOKEN_PAD="<p>"
TOKEN_START="<s>"
TOKEN_END="</s>"

ID_PAD=0
ID_START=1
ID_END=2

source_sentences = ["취미가 뭐예요", "만나서 반가워요", "내일 만나서 놀아요"]
target_sentences = ["what is your hobby", "nice to meet you", "meet tomorrow"]

source_vocab = [TOKEN_PAD, TOKEN_START, TOKEN_END, "취미가", "뭐예요", "만나서", "반가워요", "내일", "놀아요"]
target_vocab = [TOKEN_PAD, TOKEN_START, TOKEN_END, "what", "is", "your", "hobby", "nice", "to", "meet", "you", "tomorrow"]

source_input_idx = [
    [3,  4,  0,  0,  0], 
    [5,  6,  0,  0,  0],  
    [7,  5,  8,  0,  0]]
target_input_idx = [
    [1,  3,  4,  5,  6], 
    [1,  7,  8,  9, 10], 
    [1,  9, 11,  0,  0]]
target_output_idx = [
    [3,  4,  5,  6,  2], 
    [7,  8,  9, 10,  2], 
    [9, 11,  2,  0,  0]]
    
source_vocab_size = len(source_vocab)
target_vocab_size = len(target_vocab)

embedding_size = 12
num_units = 12

encoder_num_layer = 3
decoder_num_layer = 3

batch_size = 3
learning_rate = 0.0001

training_steps = 40000
display_step = 200

max_sentence_length = 5

beam_width = 3
mode = "infer"

source_inputs = tf.placeholder(dtype=tf.int64, shape=(None, max_sentence_length), name='source_inputs')
target_inputs = tf.placeholder(dtype=tf.int64, shape=(None, max_sentence_length), name='target_inputs')
target_outputs = tf.placeholder(dtype=tf.int64, shape=(None, max_sentence_length), name='target_outputs')

sequence_lengths = [max_sentence_length] * batch_size

def build_single_cell(num_units):
    cell = tf.contrib.rnn.BasicLSTMCell(num_units)
    return cell
                   
with tf.variable_scope('encoder'):
    
    initializer = tf.contrib.layers.xavier_initializer()
    embedding_encoder = tf.get_variable(name="embedding_encoder",
                                        shape=[source_vocab_size, embedding_size], 
                                        dtype=tf.float32,
                                        initializer=initializer,
                                        trainable=True)
    
    encoder_embeddding_inputs = tf.nn.embedding_lookup(params=embedding_encoder,
                                                       ids=source_inputs)
    
    # Bidirectional rnn #
    forward_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
    backward_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

    bi_outputs, encoder_state = tf.nn.bidirectional_dynamic_rnn(forward_cell, 
                                                                backward_cell, 
                                                                encoder_embeddding_inputs,
                                                                dtype=tf.float32)
    encoder_outputs = tf.concat(bi_outputs, -1)
    ######################
    
    encoder_cell_list = [build_single_cell(num_units) for i in range(encoder_num_layer-2)]
    encoder_multi_cell = tf.contrib.rnn.MultiRNNCell(encoder_cell_list)
    
    encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(cell=encoder_multi_cell,
                                                             inputs=encoder_outputs,
                                                             dtype=tf.float32)

        
with tf.variable_scope('decoder'):
    
    initializer = tf.contrib.layers.xavier_initializer()
    embedding_decoder = tf.get_variable(name="embedding_decoder",
                                        shape=[target_vocab_size, embedding_size], 
                                        dtype=tf.float32,
                                        initializer=initializer,
                                        trainable=True)
    
    decoder_embeddding_inputs = tf.nn.embedding_lookup(params=embedding_decoder,
                                                       ids=target_inputs)

    # Beam search #
    if mode == "train":
        decoder_initial_state = encoder_final_state
        beam_batch_size = batch_size
    if mode == "infer":
        decoder_initial_state = tf.contrib.seq2seq.tile_batch(encoder_final_state, multiplier=beam_width)
        encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=beam_width)
        sequence_lengths = sequence_lengths * beam_width
        beam_batch_size = batch_size * beam_width

    ###############

    decoder_cell_list = [build_single_cell(num_units) for i in range(decoder_num_layer)]
        
    # Attention mechanism #
    attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=num_units, 
                                                            memory=encoder_outputs,
                                                            memory_sequence_length=sequence_lengths) 
    ######################
    
    # Attention mechanism #
    for idx in range(len(decoder_cell_list)):
        decoder_cell_list[idx] = tf.contrib.seq2seq.AttentionWrapper(
                cell=decoder_cell_list[idx],
                attention_mechanism=attention_mechanism,
                attention_layer_size=num_units,
                name='Attention_Wrapper')
    ######################
    
    decoder_multi_cell = tf.contrib.rnn.MultiRNNCell(decoder_cell_list)
        
    # Attention mechanism #
    decoder_initial_state = decoder_multi_cell.zero_state(batch_size=beam_batch_size, dtype=tf.float32)
    decoder_initial_state = tuple(decoder_initial_state)
    ######################
    


    projection_layer = tf.layers.Dense(target_vocab_size, use_bias=False)
    
    # Beam search #
    if mode == "train":
        helper = tf.contrib.seq2seq.TrainingHelper(decoder_embeddding_inputs, sequence_lengths)
        decoder = tf.contrib.seq2seq.BasicDecoder(decoder_multi_cell, helper, decoder_initial_state, output_layer=projection_layer)
        decoder_outputs, decoder_final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder)

        decoder_predict = decoder_outputs.sample_id
        decoder_outputs = decoder_outputs.rnn_output
        
    if mode == "infer":
        start_tokens = tf.fill([batch_size], ID_START)
        end_token = ID_END
        decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                                        cell=decoder_multi_cell,
                                        embedding=embedding_decoder,
                                        start_tokens=start_tokens,
                                        end_token=end_token,
                                        initial_state=decoder_initial_state,
                                        beam_width=beam_width,
                                        output_layer=projection_layer,
                                        length_penalty_weight=0.0)
        
        decode_max_length = max_sentence_length * 2
        decoder_outputs, decoder_final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,
                                                                                   maximum_iterations = decode_max_length)
        
        decoder_predict = decoder_outputs.predicted_ids
        decoder_predict = tf.transpose(decoder_predict, perm=[0, 2, 1]) #[batch_size, beam_width, time]
        
    ###############
    
    
with tf.variable_scope("optimizer"):
    
    if mode == "train":
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=decoder_outputs,
                                                                labels=target_outputs)
        cost = tf.reduce_mean(cross_entropy)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        optimizer = optimizer.minimize(cost)

        correct_pred = tf.equal(tf.argmax(decoder_outputs, 2), target_outputs)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    

# Start training
with tf.Session() as sess:
    
    # Saver
    saver = tf.train.Saver()
    
    if mode == "train":
        # Run the initializer
        sess.run(tf.global_variables_initializer())
    
        for step in range(1, training_steps + 1):
            batch_source_input = source_input_idx
            batch_target_input = target_input_idx
            batch_target_output = target_output_idx

            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={source_inputs: batch_source_input, 
                                           target_inputs: batch_target_input,
                                           target_outputs: batch_target_output})

            if step % display_step == 0 or step == 1:
                # Calculate batch accuracy & loss
                outputs, acc, loss = sess.run([decoder_predict, accuracy, cost], feed_dict={source_inputs: batch_source_input, 
                                                                  target_inputs: batch_target_input,
                                                                  target_outputs: batch_target_output})

                print("Step " + str(step * batch_size) + ", Minibatch Loss= " +                       "{:.6f}".format(loss) + ", Training Accuracy= " +                       "{:.5f}".format(acc))

            if acc >= 1.0:
                for sentence in outputs:
                    sentence = [target_vocab[word_idx] for word_idx in sentence]
                    print("           -> ", sentence)
                    
                saver.save(sess, './save/beamsearch.ckpt')

                break;

        print("Optimization Finished!")
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={source_inputs: source_input_idx, 
                                                                 target_inputs: target_input_idx,
                                                                 target_outputs: target_output_idx}))

    if mode == "infer":
        # Restore
        saver.restore(sess, './save/beamsearch.ckpt')
        
        batch_source_input = source_input_idx
        outputs = sess.run(decoder_predict, feed_dict={source_inputs: batch_source_input})
        
        for idx, output in enumerate(outputs):
            source_sentence = [source_vocab[word_idx] for word_idx in batch_source_input[idx]]
            print(" ".join(source_sentence))
            for sentence in output:
                sentence = [target_vocab[word_idx] for word_idx in sentence]
                print("           -> ", " ".join(sentence))
            print("\n")

