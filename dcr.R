#Copyright notice

#Please remember that this package is provided to you by www.deepcreditrisk.com and its authors Harald Scheule and Daniel Roesch. The package is protected by copyright. You are not permitted to re-use for commercial purposes without permission of the copyright owner. Improper or illegal use may lead to prosecution for copyright infringement. 

#The module provides the package references for the functions used in book "Scheule, H./ Roesch, D.: Deep Credit Risk - Machine learning in R, 2022":

#packages and basic settings
#dataprep
#woe
#validation
#resolutionbias

#packages and basic settings
defaultW <- getOption("warn") 
options(warn = -1) 

suppressMessages(library(ada))
suppressMessages(library(adabag))
suppressMessages(library(AssetCorr))
suppressMessages(library(bestglm))
suppressMessages(library(betareg))
suppressMessages(library(caret))
suppressMessages(library(clue))
suppressMessages(library(data.table))
suppressMessages(library(DescTools))
suppressMessages(library(dummies))
suppressMessages(library(e1071))
suppressMessages(library(fastDummies))
suppressMessages(library(flexclust))
suppressMessages(library(FNN))
suppressMessages(library(forecast))
suppressMessages(library(glmnet))
suppressMessages(library(grid))
suppressMessages(library(gridExtra))
suppressMessages(library(gtable))
suppressMessages(library(h2o))
suppressMessages(library(JOUSBoost))
suppressMessages(library(knitr))
suppressMessages(library(lightgbm))
suppressMessages(library(lmtest))
suppressMessages(library(MLmetrics))
suppressMessages(library(mvtnorm))
suppressMessages(library(nloptr))
suppressMessages(library(nnet))
suppressMessages(library(pec))
suppressMessages(library(pROC))
suppressMessages(library(ROCR))
suppressMessages(library(rpart))
suppressMessages(library(rpart.plot))
suppressMessages(library(tidyverse))
suppressMessages(library(timereg))
suppressMessages(library(tseries))
suppressMessages(library(urca))
suppressMessages(library(xgboost))


data <- read_csv("dcr.csv", col_types = cols())

#function dataprep

dataprep <- function(data_in, depvar = "default_time", splitvar = "time", threshold = 26) {
  
  df <- data_in %>% 
        drop_na(all_of(c("time", "default_time","LTV_time", "FICO_orig_time"))) %>% 
        mutate(annuity= interest_rate_time/(100*4)*balance_orig_time/(1-(1+interest_rate_time/(100*4))**(-(mat_time-orig_time))), balance_scheduled_time = balance_orig_time*(1+ interest_rate_time/(100*4))**(time - orig_time)- annuity*((1+ interest_rate_time/(100*4))**(time - orig_time)-1)/(interest_rate_time/(100*4)), 
               property_orig_time = balance_orig_time/(LTV_orig_time/100), 
               cep_time= (balance_scheduled_time - balance_time)/property_orig_time, 
               equity_time =1-(LTV_time/100),
               age= pmin((time - first_time + 1), 40), 
               age_1 = age -1, 
               age_1f = pmax(age_1, 1), 
               age2 = age**2, 
               vintage = pmax(pmin(orig_time, 30),0), 
               state_orig_time= factor(state_orig_time)  ) %>% 
        drop_na(all_of(c("time", "cep_time","equity_time")))
    
    #Economic Features
  
  if (depvar == "default_time") {
    df2 <- df
    df2 <- df2 %>% 
      drop_na(state_orig_time)  
    small_states <- c("VI", "DC", "PR", "NA", "AL", "AK", "AR", "ND", "SD", "MT", "DE", "WV", "VT", "ME", "NE", "NH", "MS")
    df2 <- df2 %>% 
      filter(!(state_orig_time %in% small_states))
   
    #Splitting
    data_train <- df2[which(df2$time < threshold+1),]
    data_test <- df2[which(df2$time > threshold),]
      
    #PCA
    suppressMessages(defaultrates_states_train <- data_train %>% group_by(time, state_orig_time,.drop =F) %>% 
      summarize(default_time = mean(default_time)) %>% 
      spread(state_orig_time, default_time) %>%
      select(time, sort(names(.))) %>%
      rename_at(vars(-time), function(x) paste0("defaultrate_", x)) %>%
      replace(is.na(.),0))
    
    suppressMessages(defaultrates_states <- df2 %>% group_by(time, state_orig_time, .drop = F) %>% 
      summarize(default_time = mean(default_time)) %>% 
      spread(state_orig_time, default_time) %>%
      select(time, sort(names(.))) %>%
      rename_at(vars(-time), function(x) paste0("defaultrate_", x)) %>%
      replace(is.na(.),0))
    
    defaultrates_states_train1 <- scale(defaultrates_states_train) %>% replace(!is.finite(.),0) 
    
    defaultrates_states1 <- scale(defaultrates_states, 
                                  center =attr(defaultrates_states_train1, "scaled:center"),
                                  scale = attr(defaultrates_states_train1, "scaled:scale")) %>% 
      replace(!is.finite(.),0)
    
    PCAdefaultrates_states_train1 <- as.data.frame(defaultrates_states_train1)
    pctrain1 <- prcomp(PCAdefaultrates_states_train1,center = TRUE, scale =FALSE)
    z_train <- as.data.frame(pctrain1$x[,1:5])
    z <- as.data.frame(predict(pctrain1, newdata = as.data.frame(defaultrates_states1)))
    z <- z[,1:5]
    
    Z_train <- z_train %>%
      rename(PCA1 = PC1, PCA2 = PC2, PCA3 = PC3, PCA4 = PC4, PCA5 = PC5)
    Z <- z %>%
      rename(PCA1 = PC1, PCA2 = PC2, PCA3 = PC3, PCA4 = PC4, PCA5 = PC5) 
    
    Z_train_1 <- Z_train %>%
      mutate_all(lag) %>% 
      rename_all(function(x) paste0(x, "_1"))               
    Z_1 <- Z %>%
      mutate_all(lag) %>% 
      rename_all(function(x) paste0(x, "_1"))
    
    defaultrates_states_train2 <- as.data.frame(cbind(defaultrates_states_train$time, Z_train_1)) %>% 
      drop_na() %>%
      setnames("defaultrates_states_train$time", "time")
    defaultrates_states2 <- as.data.frame(cbind(defaultrates_states$time, Z_1)) %>% 
      drop_na() %>%
      setnames("defaultrates_states$time", "time")
    
    df3_tmp <- merge(x = df2, y = defaultrates_states2, by = "time", all.x=TRUE)
    df3 <- df3_tmp[with(df3_tmp, order(id, time)),]      
    df3 <- df3 %>% select(id, everything())

    data_train_tmp <- merge(x = data_train, y = defaultrates_states_train2, by = "time", all.x=TRUE)
    data_train <- data_train_tmp[with(data_train_tmp, order(id, time)),]  
    data_train <- data_train %>% select(id, everything())
    
    data_test_tmp <- merge(x = data_test, y = defaultrates_states2, by = "time", all.x=TRUE)
    data_test <- data_test_tmp[with(data_test_tmp, order(id, time)),]  
    data_test <- data_test %>% select(id, everything())
    
    #Scaling
    df3 <- df3[complete.cases(df3[, c("cep_time", "equity_time", "interest_rate_time", "FICO_orig_time", "gdp_time", "PCA1_1", "PCA2_1", "PCA3_1", "PCA4_1", "PCA5_1")]),]
    data_train <- data_train[complete.cases(data_train[, c("cep_time", "equity_time", "interest_rate_time", "FICO_orig_time", "gdp_time", "PCA1_1", "PCA2_1", "PCA3_1", "PCA4_1", "PCA5_1")]),]         
    data_test <- data_test[complete.cases(data_test[, c("cep_time", "equity_time", "interest_rate_time", "FICO_orig_time", "gdp_time", "PCA1_1", "PCA2_1", "PCA3_1", "PCA4_1", "PCA5_1")]),]         

    X_train <- subset(data_train, select =c("cep_time", "equity_time", "interest_rate_time", "FICO_orig_time", "gdp_time", "PCA1_1", "PCA2_1", "PCA3_1", "PCA4_1", "PCA5_1"))    
    X_test <- subset(data_test, select =c("cep_time", "equity_time", "interest_rate_time", "FICO_orig_time", "gdp_time", "PCA1_1", "PCA2_1", "PCA3_1", "PCA4_1", "PCA5_1"))
 
    
    X_train_scaled <- scale(X_train)
    X_test_scaled <- scale(X_test,center =attr(X_train_scaled, "scaled:center"), scale = attr(X_train_scaled, "scaled:scale")) 
    
    y_train <- as.vector(data_train$default_time)
    y_test <- as.vector(data_test$default_time)
    
    #Clustering 
    set.seed(2)
    n_clusters <- 2
    kmeans <- kmeans(X_train_scaled, centers = n_clusters)
   
    clusters_train <- as.data.frame(as.matrix(cl_predict(kmeans, X_train_scaled)))                 
    clusters_test <- as.data.frame(as.matrix(cl_predict(kmeans, X_test_scaled)))
                 
    dummies_train <- dummy_cols(clusters_train,remove_first_dummy = TRUE)[-1]
    dummies_test <- dummy_cols(clusters_test, remove_first_dummy = TRUE)[-1]
    
    colnames(dummies_train) <- c("cluster_1")
    colnames(dummies_test) <- c("cluster_1")       
    
    X_train_scaled <- cbind(X_train_scaled, dummies_train) 
    X_test_scaled <- cbind(X_test_scaled, dummies_test)

    X_train_scaled <- as.matrix(X_train_scaled) 
    X_test_scaled <- as.matrix(X_test_scaled)
                 
    dummies <- rbind(dummies_train, dummies_test)

    df3 <- cbind(df3, dummies)
    data_train <- cbind(data_train, dummies_train)
    data_test <- cbind(data_test,dummies_test)    
  }
  
  if (depvar == "lgd_time") {
    
    #LGD dataprep
    df2 <- df %>% 
    filter(default_time == 1)
    df3 <- resolutionbias(df2, lgd = "lgd_time", res = "res_time", t = "time")
    
    df3 <- df3 %>% 
    mutate(lgd_time = pmin(pmax(lgd_time, 0.0001), 0.9999)) 
      
    data_train <- df3 %>% 
      filter(time < threshold+1)
    data_test <- df3 %>% 
      filter(time > threshold)
    
    X_train <- subset(data_train, select = c("cep_time", "equity_time", "interest_rate_time", "FICO_orig_time", "REtype_CO_orig_time", "REtype_PU_orig_time", "gdp_time"))
    X_test <- subset(data_test, select = c("cep_time", "equity_time", "interest_rate_time", "FICO_orig_time", "REtype_CO_orig_time", "REtype_PU_orig_time", "gdp_time"))
    
    y_train <- as.vector(data_train$lgd_time)
    y_test <- as.vector(data_test$lgd_time)
    
    X_train_scaled <- scale(X_train)
    X_test_scaled <- scale(X_test,center =attr(X_train_scaled, "scaled:center"), scale = attr(X_train_scaled, "scaled:scale")) 

    dummies_train <- dummy_cols(data_train$state_orig_time, remove_first_dummy = TRUE,ignore_na = T)[-1]
    dummies_test <- dummy_cols(data_test$state_orig_time, remove_first_dummy = TRUE,ignore_na = T)[-1]
      
    X_train_scaled <- cbind(X_train_scaled, dummies_train) 
    X_test_scaled <- cbind(X_test_scaled, dummies_test)

    X_train_scaled <- as.matrix(X_train_scaled) 
    X_test_scaled <- as.matrix(X_test_scaled)
    
    dummies <- rbind(dummies_train, dummies_test)

  }
  data <- df3
  my_list <- list("data" = data, "data_train" = data_train, "data_test" = data_test, "X_train_scaled" = X_train_scaled, "X_test_scaled" = X_test_scaled, "y_train" = y_train, "y_test" = y_test)
  
  return (my_list)
  
}

#function woe
woe <- function(data_in, target, variable, bins, binning) {
    
    df <- data_in
    df2 <- na.omit(subset(data_in, select = c(target, variable)))
    colnames(df2) <- c("Target", "Variable")
    
    if (binning == TRUE) {
        df2$key <- cut(df2$Variable, unique(quantile(df2$Variable,probs = seq(0,1,1/bins), 
                                                na.rm = T)))
    }
    if (binning == FALSE) {
        df2$key <- df2$Variable
    }
    
    table <- table(df2$key, df2$Target)
    All <- margin.table(table,1)
    table <- as.data.frame(cbind(table, All))
    colnames(table) <- c("nondeft", "deft", "All")
    setDT(table, keep.rownames = "key")[]
    
    table$fracdeft <- table$deft /sum(table$deft)
    table$fracnondeft <- table$nondeft / sum(table$nondeft)
    
    table$WOE <- log(table$fracdeft/table$fracnondeft)
    table$IV <- (table$fracdeft-table$fracnondeft)*table$WOE
    colnames(table)[7] <- variable
    table <- table %>%
        setNames(c(names(.)[1], paste0(names(.)[-1], "_WOE")))
    
    WOE <- table[, c(1,7)]
    
    df <- merge(df, df2$key, by = 0, all = FALSE)[-1]
    
    names(df)[names(df) == "y"] <- "key"
    
    outputWOE <- merge(df, WOE, by = "key")
    outputWOE$key <- NULL
    outputIV <- data.frame("Variable" = variable, "IV" = sum(table$IV_WOE))
    
    return (list("outputWOE" = outputWOE, "outputIV" = outputIV))
    
}             
                                 
# function :=
':=' <- function(lhs, rhs) {
  frame <- parent.frame()
  lhs <- as.list(substitute(lhs))
  if (length(lhs) > 1)
    lhs <- lhs[-1]
  if (length(lhs) == 1) {
    do.call(`=`, list(lhs[[1]], rhs), envir=frame)
    return(invisible(NULL)) 
  }
  if (is.function(rhs) || is(rhs, 'formula'))
    rhs <- list(rhs)
  if (length(lhs) > length(rhs))
    rhs <- c(rhs, rep(list(NULL), length(lhs) - length(rhs)))
  for (i in 1:length(lhs))
    do.call(`=`, list(lhs[[i]], rhs[[i]]), envir=frame)
  return(invisible(NULL)) 
}
                 
#function validation   
validation <- function(fit, outcome, time, continuous = FALSE) {
    options(repr.plot.width = 16, repr.plot.height = 9, repr.plot.res = 300)
    
    fit <- as.vector(as.numeric(unlist(fit)))
    outcome <- as.vector(as.numeric(unlist(outcome)))
    time <- as.vector(as.numeric(unlist(time)))

    fitP <- data.frame(fit)
    outcomeP <- data.frame(outcome)
    timeP <- data.frame(time)
   
    data_in <- cbind(fitP, outcomeP, timeP)
    colnames(data_in) <- c("fit", "outcome", "time")

    data_in$outcomeD <- data_in$outcome
    
    if(continuous == TRUE) {
        data_in[data_in$outcome >= mean(data_in$outcome), "outcomeD"] <- 1
        data_in[data_in$outcome <= mean(data_in$outcome), "outcomeD"] <- 0
    }
    
    outcomeD <- data_in$outcomeD
    
    lr_log_loss <- NA
    roc_auc <- NA
    brier <- NA
    binom_p <- NA
    Jeffreys_p <- NA
    
    max_outcome_fit <- max(max(outcome), max(fit))
    min_outcome_fit <- min(min(outcome), min(fit))
    
    if (min_outcome_fit >= 0 & max_outcome_fit <=1) {
    lr_log_loss <- round(LogLoss(fit,outcomeD), digits = 4)
    roc_auc <- suppressMessages(round(auc(outcomeD, fit), digits = 4))
    binom_p <- round(binom.test(sum(outcomeD), n = length(outcomeD), p =mean(fit), alternative = c("greater"))$p.value, digits=4)
    Jeffreys_p <- round(pbeta(mean(fit), sum(outcomeD)+ 0.5, length(outcomeD)-sum(outcomeD)+0.5), digits =4)
    }
    
    corr <- cor(fit, outcome, method = c("pearson"))
    r2_OLS <- corr**2
    
    the_table <- c("Counts" = length(outcome),
                       "Mean outcome" = round(sum(outcome)/length(outcome), digits=4),
                       "Mean fit" = round(mean(fit), digits = 4),
                       "AUC" = roc_auc,
                       "R-squared (OLS)" = round(r2_OLS, digits =4),
                       "R-squared" = round(R2_Score(fit, outcome), digits=4),
                       "RMSE/SQR(Brier score)" = round(sqrt(((outcome-fit)%*%(outcome-fit))/length(outcome)), digits =4),
                       "Log Loss" = lr_log_loss,
                       "Binomial p-value" = binom_p,
                       "Jeffreys p-value" = Jeffreys_p)
    the_table <- as.data.frame(the_table)
    the_table <- setDT(the_table, keep.rownames = TRUE)[]
    the_table <- tibble(Metric=the_table[,1], Value=as.character(as.numeric(unlist(the_table[,2]))))
    
    tt1 <- ttheme_minimal(core = list(fg_params=list(fontface="plain",cex=1.1,just="centre"),bg_params=list(col="black",lwd=2)), colhead = list(fg_params=list(col="black",fontface="bold",cex=1.1,just="centre"),bg_params=list(col="black",lwd=2)))
    table <- tableGrob(the_table, rows = NULL, theme = tt1)
    title <- textGrob("Summary", gp=gpar(fontsize=30))
    table <- gtable_add_rows(table, heights=grobHeight(title) + unit(13, "mm"), pos = 0)
    table <- gtable_add_rows(table, unit(13, "mm"))
    table <- gtable_add_grob(table, title, 1, 1, 1, ncol(table))
    
    p1 <- data_in %>% 
        dplyr::select("fit", "outcome", "time") %>% 
        group_by(time) %>% 
        summarise(obs=mean(outcome), 
                  pred=mean(fit)) %>% 
        ggplot(aes(x=time)) + 
        ggtitle("Time-Series Real Fit") + 
        #xlim(min_outcome_fit,max_outcome_fit) + 
        geom_line(aes(y=obs, linetype="Outcome", color="Outcome"), size=1) +  
        geom_line(aes(y=pred, linetype="Fit", color="Fit"), size=1) + 
        scale_linetype_manual(name=NULL, values = c("Outcome" = 1, "Fit" = 2)) +
        scale_color_manual(name=NULL, values = c("Outcome" = "blue", "Fit" = "red")) +
        labs(x="Time", y="Mean") +
        theme_bw(base_size = 25) + 
        theme(legend.position = c(0.8, 0.8), legend.background=element_rect(fill = alpha("white", 0)),
      legend.key=element_rect(fill = alpha("white", .5))) +
        theme(plot.title = element_text(hjust = 0.5)) %>% 
        suppressMessages()

    p2 <- data_in %>% ggplot(aes(x=fit)) + 
        ggtitle("Fit Histogram") + 
        geom_histogram(bins=50, color="black", fill="blue") + 
        labs(x="Fit", y="Frequency") +
        theme_bw(base_size = 25) +
        theme(plot.title = element_text(hjust = 0.5)) %>% 
        suppressMessages()

    data_in$cat <- cut(data_in$fit, unique(quantile(as.numeric(data_in$fit),probs = seq(0,1,0.1), na.rm = T)), include.lowest = TRUE, labels = FALSE)

    p3 <- data_in %>%
        group_by(cat) %>%
        summarise(obs=mean(outcome), 
              pred=mean(fit))

        maximum=max(max(p3$obs), max(p3$pred))       
        maximum=ceiling(maximum*100)/100
        minimum=min(min(p3$obs), min(p3$pred))
        minimum=floor(minimum*100)/100

    p3 <- p3 %>% 
        ggplot(aes(x=pred, y=obs)) + 
        xlim(minimum,maximum) + 
        ylim(minimum,maximum) + 
        ggtitle("Calibration Curve") + 
        geom_point(size=5, color="blue") +
        geom_abline(slope=1, intercept=0, color="red", size=1) + 
        labs(x="Mean fit", y="Mean outcome") +
        theme_bw(base_size = 25) + 
        theme(plot.title = element_text(hjust = 0.5)) %>% 
        suppressMessages()
grid.arrange(table, p1, p2, p3, nrow=2, ncol=2)
}
                 
#function resolutionbias
resolutionbias <- function(data_in, lgd, res, t) {
  
  df <- data_in
  
  df1 <- df %>% 
    filter(is.na(res_time)==FALSE) %>% 
    mutate(res_period= res_time - time) 

  df2 <- df1 %>% 
    mutate(res_period = pmin(res_period, 20)) 
  
  data_LGD_sum <- df2%>% 
    group_by(res_period) %>% 
    summarize(lgd_time = sum(lgd_time))%>%
    arrange(desc(res_period))
  
  data_LGD_count <- df2 %>% 
    group_by(res_period) %>% 
    summarize(lgd_time = n()) %>%
    arrange(desc(res_period)) 
  
  data_LGD_sum_cumsum <- cumsum(data_LGD_sum)

  data_LGD_count_cumsum <- cumsum(data_LGD_count)

  data_LGD_mean <- data_LGD_sum_cumsum/data_LGD_count_cumsum 
  
  data_LGD_mean <- data_LGD_mean %>%   
    mutate(res_period = data_LGD_sum$res_period)
  
  data_LGD_mean2 <- data_LGD_mean[rep(seq_len(nrow(data_LGD_mean[1,])), each = 41),]

  data_LGD_mean3 <- rbind(data_LGD_mean2, data_LGD_mean)

  n  <- nrow(data_LGD_mean3)-1

  data_LGD_mean3 <- data_LGD_mean3 %>%
    mutate(time=0:n) %>%
    relocate(time, .before = lgd_time)

  df_replace <- df %>% 
    filter(is.na(res_time)==TRUE) %>% 
    select(-c(lgd_time))

  df_replace2 <- left_join(df_replace, data_LGD_mean3, by ="time")

  df2 <- df2 %>% 
    select(colnames(df_replace2))
           
  df3 <- rbind(df2, df_replace2)

  df3 <- df3 %>% 
    select(-res_period)
    
  return (df3)
}
options(warn = defaultW)
