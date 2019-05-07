# Bag-of-Embedded Words Model

We provide a python implementation of the [Bag-of-Embedded-Words (BoEW) method](https://passalis.github.io/assets/pdf/papers/j2.pdf). The BoEW method employs the well known Bag-of-Features model to provide a way to learn compact document-level representations using word-embedding models.  The method was implemented using the theano library (which is no longer supported - I hope to provide a PyTorch-based implementation soon).

If you use this code in your work please cite the following paper:

<pre>
@article{passalis2018boew,
  title={Learning Bag-of-Embedded-Words Representations for Textual Information Retrieval},
  author={Passalis, Nikolaos and Tefas, Anastasios},
  journal={Pattern Recognition},
  volume={81},
  pages={254--267},
  year={2018},
}
</pre>
