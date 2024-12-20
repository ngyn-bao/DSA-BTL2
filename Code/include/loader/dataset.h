/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/file.h to edit this template
 */

/*
 * File:   dataset.h
 * Author: ltsach
 *
 * Created on September 2, 2024, 3:59 PM
 */

#ifndef DATASET_H
#define DATASET_H
#include "xtensor_lib.h"
using namespace std;

template <typename DType, typename LType>
class DataLabel
{
private:
    xt::xarray<DType> data;
    xt::xarray<LType> label;

public:
    DataLabel(xt::xarray<DType> data, xt::xarray<LType> label) : data(data), label(label)
    {
    }
    xt::xarray<DType> getData() const { return data; }
    xt::xarray<LType> getLabel() const { return label; }
};

template <typename DType, typename LType>
class Batch
{
private:
    xt::xarray<DType> data;
    xt::xarray<LType> label;

public:
    Batch(xt::xarray<DType> data, xt::xarray<LType> label) : data(data), label(label)
    {
    }
    virtual ~Batch() {}
    xt::xarray<DType> &getData() { return data; }
    xt::xarray<LType> &getLabel() { return label; }
};

template <typename DType, typename LType>
class Dataset
{
private:
public:
    Dataset() {};
    virtual ~Dataset() {};

    virtual int len() = 0;
    virtual DataLabel<DType, LType> getitem(int index) = 0;
    virtual xt::svector<unsigned long> get_data_shape() = 0;
    virtual xt::svector<unsigned long> get_label_shape() = 0;
};

//////////////////////////////////////////////////////////////////////
template <typename DType, typename LType>
class TensorDataset : public Dataset<DType, LType>
{
private:
    xt::xarray<DType> data;
    xt::xarray<LType> label;
    xt::svector<unsigned long> data_shape, label_shape;

public:
    /* TensorDataset:
     * need to initialize:
     * 1. data, label;
     * 2. data_shape, label_shape
     */
    TensorDataset(xt::xarray<DType> data, xt::xarray<LType> label)
    {
        /* TODO: your code is here for the initialization
         */
        this->data = data;
        this->label = label;

        this->data_shape = xt::svector<unsigned long>(this->data.shape().begin(), this->data.shape().end());
        if (label.size() == 0)
        {
            this->label_shape = xt::svector<unsigned long>{0};
        }
        else
        {
            this->label_shape = xt::svector<unsigned long>(this->label.shape().begin(), this->label.shape().end());
        }
    }
    /* len():
     *  return the size of dimension 0
     */
    int len()
    {
        /* TODO: your code is here to return the dataset's length
         */
        return this->data_shape[0];
        // return 0; // remove it when complete
    }

    /* getitem:
     * return the data item (of type: DataLabel) that is specified by index
     */
    DataLabel<DType, LType> getitem(int index)
    {
        /* TODO: your code is here
         */
        if (index < 0 || index >= this->len())
        {
            throw std::out_of_range("Index out of range.");
        }

        xt::xarray<DType> data_item = xt::view(this->data, index);
        if (this->label_shape.size() == 0 || (this->label_shape.size() == 1 && this->label_shape[0] == 0))
        {
            xt::xarray<LType> label_item = xt::zeros<LType>({1});
            return DataLabel<DType, LType>(data_item, label_item);
        }
        else
        {
            xt::xarray<LType> label_item = xt::view(this->label, index);
            return DataLabel<DType, LType>(data_item, label_item);
        }
    }

    xt::svector<unsigned long> get_data_shape()
    {
        /* TODO: your code is here to return data_shape
         */
        return this->data_shape;
    }

    xt::svector<unsigned long> get_label_shape()
    {
        if (this->label_shape.size() == 1 && this->label_shape[0] == 0)
        {
            return xt::svector<unsigned long>{0};
        }
        else
        {
            return this->label_shape;
        }
    }
};

#endif /* DATASET_H */
